import argparse

import numpy as np
import stheno.torch as stheno
import torch
import matplotlib.pyplot as plt

import lib.data
from lib.ode_rnn import get_net
from lib.losses import get_loss

from lib.utils import (
    device,
    report_loss,
    RunningAverage,
    generate_root,
    WorkingDirectory,
    save_checkpoint,
    update_learning_rate
)

plt.switch_backend('agg')

def validate(data, model, log_likelihood, mse_loss, report_freq=None):
    """Compute the validation loss."""
    losses = {'mse': mse_loss, 
              'log_likelihood': log_likelihood}
    ravg = {'mse': RunningAverage(), 
            'log_likelihood': RunningAverage(), 
            'mse_extrap': RunningAverage(),
            'log_likelihood_extrap': RunningAverage()}

    model.eval()
    with torch.no_grad():
        for step, task in enumerate(data):

            pred = model(task['y'], task['mask_obs'], task['mask_first'], task['x'], 
                extrap_time=task['extrap_time'])

            for loss in losses:
                index = task['extrap_index']
                loss_obj = losses[loss](task['y'][:, :index], pred[:, :index-1], 
                    task['mask_y'][:, :index])

                ravg[loss].update(loss_obj.item(), 1)
                if report_freq:
                    report_loss(f'Validation {loss}', ravg[loss].avg, step, report_freq)

                if index < task['x'].size()[0]:
                    loss_obj = losses[loss](task['y'][:, index-1:], pred[:, index-1:], 
                        task['mask_y'][:, index-1:])

                    ravg[f'{loss}_extrap'].update(loss_obj.item(), 1)
                    if report_freq:
                        report_loss(f'Validation {loss} extrap', ravg[f'{loss}_extrap'].avg, 
                            step, report_freq)

    return {loss: ravg[loss].avg for loss in ravg}


def train(data, model, loss, opt, use_sampling, report_freq):
    """Perform a training epoch."""
    ravg = RunningAverage()
    model.train()
    for step, task in enumerate(data):
        pred = model(task['y'], task['mask_obs'], task['mask_first'], task['x'], 
            use_sampling=use_sampling)
        obj = loss(task['y'], pred, task['mask_y'])
        obj.backward()
        opt.step()
        opt.zero_grad()
        ravg.update(obj.item(), 1)
        report_loss('Training', ravg.avg, step, report_freq)
    return ravg.avg


def to_numpy(x):
    """Convert a PyTorch tensor to NumPy."""
    return x.squeeze().detach().cpu().numpy()


def plot_model_task(model, data, epoch, wd):

    for step, task in enumerate(data):
        model.eval()
        with torch.no_grad():
            pred = to_numpy(model(task['y'], task['mask_obs'], task['mask_first'], 
                task['x'], extrap_time=task['extrap_time']))

        observations_mask = to_numpy(task['mask_obs'])
        ground_truth_data = to_numpy(task['y'])
        x = to_numpy(task['x'])

        for i in range(len(ground_truth_data)):
            # Plot context.
            fig = plt.figure()
            num_context_points = (x[observations_mask[i] == 1] <= task['extrap_time']).sum()
            plt.scatter(x[observations_mask[i] == 1][:num_context_points],
                        ground_truth_data[i, observations_mask[i] == 1][:num_context_points],
                        label='Context Set', color='indianred')
            plt.plot(x[1:], pred[i], label='Predicted', color='navy')
            plt.plot(x, ground_truth_data[i], label='Oracle GP', color='forestgreen')

            plt.legend()
            plt.savefig(wd.file('plots', f'epoch_{epoch + 1}_plot_{i + 1}.png'))
            plt.close()


# Parse arguments given to the script.
parser = argparse.ArgumentParser()
parser.add_argument('data',
                    choices=['eq',
                             'matern',
                             'noisy-mixture',
                             'weakly-periodic',
                             'sawtooth'],
                    help='Data set to train the CNP on. ')
parser.add_argument('--root',
                    help='Experiment root, which is the directory from which '
                         'the experiment will run. If it is not given, '
                         'a directory will be automatically created.')
parser.add_argument('--train',
                    action='store_true',
                    help='Perform training. If this is not specified, '
                         'the model will be attempted to be loaded from the '
                         'experiment root.')
parser.add_argument('--epochs',
                    default=200,
                    type=int,
                    help='Number of epochs to train for.')
parser.add_argument('--learning_rate',
                    default=1e-2,
                    type=float,
                    help='Learning rate.')
parser.add_argument('--weight_decay',
                    default=1e-5,
                    type=float,
                    help='Weight decay.')

# Model specification
parser.add_argument('--latent_dim',
                    default=10,
                    type=int)
parser.add_argument('--ode_func_layers',
                    default=1,
                    type=int)
parser.add_argument('--ode_func_units',
                    default=100,
                    type=int)
parser.add_argument('--input_dim',
                    default=1,
                    type=int)
parser.add_argument('--decoder_units',
                    default=100,
                    type=int)
parser.add_argument('--extrapolation',
                    action='store_true')
parser.add_argument('--use_sampling',
                    action='store_true')

args = parser.parse_args()

# Load working directory.
if args.root:
    wd = WorkingDirectory(root=args.root)
else:
    experiment_name = f'ode_rnn-{args.data}'
    wd = WorkingDirectory(root=generate_root(experiment_name))

# Load data generator.
if args.data == 'sawtooth':
    gen = lib.data.SawtoothGenerator()
    gen_val = lib.data.SawtoothGenerator(num_tasks=60, extrapolation=args.extrapolation)
    gen_test = lib.data.SawtoothGenerator(num_tasks=2048, extrapolation=args.extrapolation)
    gen_plot = lib.data.SawtoothGenerator(num_tasks=1, batch_size=3, plot=True, 
        extrapolation=args.extrapolation)
else:
    if args.data == 'eq':
        kernel = stheno.EQ().stretch(0.25)
    elif args.data == 'matern':
        kernel = stheno.Matern52().stretch(0.25)
    elif args.data == 'noisy-mixture':
        kernel = stheno.EQ().stretch(1.) + \
                 stheno.EQ().stretch(.25) + \
                 0.001 * stheno.Delta()
    elif args.data == 'weakly-periodic':
        kernel = stheno.EQ().stretch(0.5) * stheno.EQ().periodic(period=0.25)
    else:
        raise ValueError(f'Unknown data "{args.data}".')

    gen = lib.data.GPGenerator(kernel=kernel)
    gen_val = lib.data.GPGenerator(kernel=kernel, num_tasks=60, extrapolation=args.extrapolation)
    gen_test = lib.data.GPGenerator(kernel=kernel, num_tasks=2048, extrapolation=args.extrapolation)
    gen_plot = lib.data.GPGenerator(kernel=kernel, num_tasks=1, batch_size=3, plot=True, 
        extrapolation=args.extrapolation)

# Load model.
model = get_net(args).to(device)

log_likelihood, mse_loss = get_loss(args)

# Perform training.
# opt = torch.optim.Adamax(model.parameters(),
#                          args.learning_rate, 
#                          weight_decay=args.weight_decay)

opt = torch.optim.Adamax(model.parameters(),
                         args.learning_rate)

print(f'Number of trainable parameters: {model.num_params}')

if args.train:
    # Run the training loop, maintaining the best objective value.
    best_obj = np.inf
    for epoch in range(args.epochs):
        print('\nEpoch: {}/{}'.format(epoch + 1, args.epochs))

        # Compute training objective.
        train_obj = train(gen, model, log_likelihood, opt, 
            use_sampling=args.use_sampling, report_freq=50)
        report_loss('Training', train_obj, 'epoch')

        # Compute validation objective.
        val_dict = validate(gen_val, model, log_likelihood, mse_loss, report_freq=20)

        report_loss('Validation', val_dict['log_likelihood'], 'epoch')

        plot_model_task(model, gen_plot, epoch, wd)

        update_learning_rate(opt, decay_rate=0.999, lowest=args.learning_rate/10)

        # Update the best objective value and checkpoint the model.
        is_best = False
        if val_dict['log_likelihood'] < best_obj:
            best_obj = val_dict['log_likelihood']
            is_best = True
        save_checkpoint(wd,
                        {'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc_top1': best_obj,
                         'optimizer': opt.state_dict()},
                        is_best=is_best)


else:
    # Load saved model.
    load_dict = torch.load(wd.file('model_best.pth.tar', exists=True))
    model.load_state_dict(load_dict['state_dict'])

# Finally, test model on ~2000 tasks.
loss_dict = validate(gen_test, model, log_likelihood, mse_loss)
for loss in loss_dict:
    print(f'Model averages a {loss} of {loss_dict[loss]} on unseen tasks.')
    with open(wd.file(f'loss.txt'), 'w') as f:
        f.write(str(loss_dict[loss]))
