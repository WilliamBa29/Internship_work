import math

import torch
from torch.distributions import constraints

from pyro.contrib.gp.kernels.isotropic import Isotropy
from pyro.contrib.gp.kernels.kernel import Kernel
from pyro.nn.module import PyroParam

class Spectralmixture(Kernel):
    r"""
    Implementation of Periodic kernel:

        :math:`k(x,z)=\sigma^2\exp\left(-2\times\frac{\sin^2(\pi(x-z)/p)}{l^2}\right),`

    where :math:`p` is the ``period`` parameter.

    References:

    [1] `Introduction to Gaussian processes`,
    David J.C. MacKay

    :param torch.Tensor lengthscale: Length scale parameter of this kernel.
    :param torch.Tensor period: Period parameter of this kernel.
    """

    def __init__(
        self, input_dim, variance=None, lengthscale=None, period=None, active_dims=None
    ):
        super().__init__(input_dim, active_dims)

        variance = torch.tensor(1.0) if variance is None else variance
        self.variance = PyroParam(variance, constraints.positive)

        lengthscale = torch.tensor(1.0) if lengthscale is None else lengthscale
        self.lengthscale = PyroParam(lengthscale, constraints.positive)

        period = torch.tensor(1.0) if period is None else period
        self.period = PyroParam(period, constraints.positive)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self.variance.expand(X.size(0))

        if Z is None:
            Z = X
        X = self._slice_input(X)
        Z = self._slice_input(Z)
        if X.size(1) != Z.size(1):
            raise ValueError("Inputs must have the same number of features.")

        d = X.unsqueeze(1) - Z.unsqueeze(0)
        scaled_exp = torch.exp(- d**2 / (2*self.lengthscale**2))
        scaled_cos = torch.cos(d/self.period)
        return self.variance * scaled_exp*scaled_cos