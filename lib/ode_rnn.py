import torch.nn as nn
import torch
import lib.utils as utils
import numpy as np

from lib.ode_func import ODEFunc
from lib.diffeq_solver import DiffeqSolver
from lib.gru_unit import GRU_Unit

def get_net(args):
    return ODE_RNN(latent_dim = args.latent_dim,
                   ode_func_layers = args.ode_func_layers,
                   ode_func_units = args.ode_func_units,
                   input_dim = args.input_dim,
                   decoder_units = args.decoder_units)


class ODE_RNN(nn.Module):
    """Class for standalone ODE-RNN model. Makes predictions forward in time."""
    def __init__(self, latent_dim, ode_func_layers, ode_func_units, input_dim, decoder_units):
        super(ODE_RNN, self).__init__()

        ode_func_net = utils.create_net(latent_dim, latent_dim,
                                        n_layers=ode_func_layers,
                                        n_units=ode_func_units,
                                        nonlinear=nn.Tanh)

        utils.init_network_weights(ode_func_net)

        rec_ode_func = ODEFunc(ode_func_net=ode_func_net)

        self.ode_solver = DiffeqSolver(rec_ode_func, "euler", odeint_rtol=1e-3, odeint_atol=1e-4)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_units),
            nn.Tanh(),
            nn.Linear(decoder_units, input_dim*2))

        utils.init_network_weights(self.decoder)

        self.gru_unit = GRU_Unit(latent_dim, input_dim, n_units=decoder_units)

        self.latent_dim = latent_dim

        self.sigma_fn = nn.Softplus()

    def forward(self, data, mask, mask_first, time_steps, extrap_time=float('inf'), use_sampling=False):

        batch_size, n_time_steps, n_dims = data.size()

        prev_hidden = torch.zeros((batch_size, self.latent_dim))
        prev_hidden_std = torch.zeros((batch_size, self.latent_dim))

        if data.is_cuda:
            prev_hidden = prev_hidden.to(data.get_device())
            prev_hidden_std = prev_hidden_std.to(data.get_device())

        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 50

        outputs = []
        outputs_std = []

        prev_observation = data[:, 0]

        if use_sampling:
            prev_output = data[:, 0]

        for i in range(1, len(time_steps)):

            # Make one step.
            if time_steps[i] - time_steps[i - 1] < minimum_step:
                inc = self.ode_solver.ode_func(time_steps[i - 1], prev_hidden)

                ode_sol = prev_hidden + inc * (time_steps[i] - time_steps[i - 1])
                ode_sol = torch.stack((prev_hidden, ode_sol), 1)
            # Several steps.
            else:
                num_intermediate_steps = max(2, ((time_steps[i] - time_steps[i - 1])/minimum_step).int())

                time_points = torch.linspace(time_steps[i - 1], time_steps[i],
                                             num_intermediate_steps)
                ode_sol = self.ode_solver(prev_hidden.unsqueeze(0), time_points)[0]

            hidden_ode = ode_sol[:, -1]

            x_i = prev_observation

            if use_sampling and np.random.uniform(0, 1) < 0.5 and time_steps[i] <= extrap_time:
                x_i = prev_output

            mask_i = mask[:, i]

            output_hidden, hidden, hidden_std = self.gru_unit(hidden_ode, prev_hidden_std,
                                                              x_i, mask_i)

            hidden = mask_first[:, i - 1] * hidden
            hidden_std = mask_first[:, i - 1] * hidden_std

            prev_hidden, prev_hidden_std = hidden, hidden_std

            mean, std = torch.chunk(self.decoder(output_hidden), chunks=2, dim=-1)

            outputs += [mean]
            outputs_std += [self.sigma_fn(std)]

            if use_sampling:
                prev_output = prev_output*(1 - mask_i) + mask_i*outputs[-1]

            if time_steps[i] <= extrap_time:
                prev_observation = prev_observation*(1 - mask_i) + mask_i*data[:, i]
            else:
                prev_observation = prev_observation*(1 - mask_i) + mask_i*outputs[-1]

        outputs = torch.stack(outputs, 1)
        outputs_std = torch.stack(outputs_std, 1)

        return outputs, outputs_std

    @property
    def num_params(self):
        """Number of parameters."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])
