# Import packages
import torch
import numpy as np

import normflows as nf

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from tqdm import tqdm
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


# Set up target
class GaussianVonMises(nf.distributions.Target):
    def __init__(self):
        super().__init__(
            prop_scale=torch.tensor(2 * np.pi), prop_shift=torch.tensor(-np.pi)
        )
        self.n_dims = 2
        self.max_log_prob = -1.99
        self.log_const = -1.5 * np.log(2 * np.pi) - np.log(np.i0(1))

    def log_prob(self, x):
        return -0.5 * x[:, 0] ** 2 + torch.cos(x[:, 1] - 3 * x[:, 0]) + self.log_const


if __name__ == "__main__":
    target = GaussianVonMises()
    # Plot target
    grid_size = 300
    xx, yy = torch.meshgrid(
        torch.linspace(-2.5, 2.5, grid_size), torch.linspace(-np.pi, np.pi, grid_size)
    )
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)

    log_prob = target.log_prob(zz).view(*xx.shape)
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0

    plt.figure(figsize=(15, 15))
    plt.pcolormesh(yy, xx, prob.data.numpy(), cmap="coolwarm")
    plt.gca().set_aspect("equal", "box")
    plt.show()
