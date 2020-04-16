import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from lib.utils import device

def get_loss(args):
    return GaussianLogLikelihoodLoss(), MSELoss()

class GaussianLogLikelihoodLoss(nn.Module):
    def __init__(self, obsrv_std=0.01):
        super(GaussianLogLikelihoodLoss, self).__init__()

        self.obsrv_std = torch.Tensor([obsrv_std]).to(device)

    def forward(self, ground_truth, pred, pred_std, mask):
        gaussian = Normal(loc=pred, scale=pred_std)
        log_prob = gaussian.log_prob(ground_truth[:, 1:])
        log_prob = (log_prob * mask[:, 1:]).sum()/mask[:, 1:].sum()
        return -log_prob


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, ground_truth, pred, pred_std, mask):
        mse_loss = mask[:, 1:] * (ground_truth[:, 1:] - pred)**2
        return mse_loss.sum()/mask[:, 1:].sum()
