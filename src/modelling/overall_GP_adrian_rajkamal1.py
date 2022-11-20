import os
import matplotlib.pyplot as plt
import torch

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

"""
    (Initial/Basic) GP for spot prices based on time.
    
    Author: Adrian Rajkamal
    Date: 16/12/2021
"""



#df = pd.read_csv("../../data/processed/historical_spot_price.csv")

""" HELPER FUNCTIONS """


def plot(plot_observed_data=False, plot_predictions=False, n_prior_samples=0, model=None,
         x_test=None):
    """ Plotting code for GP posterior - from GP Pyro tutorial"""
    n_test = len(x_test)
    plt.figure(figsize=(12, 6))
    if plot_observed_data:
        plt.plot(X.numpy(), y.numpy(), 'kx')
    if plot_predictions:
        if x_test is None:
            x_test = torch.linspace(-0.5, 5.5, n_test)  # test inputs
        # compute predictive mean and variance
        with torch.no_grad():
            if type(model) == gp.models.VariationalSparseGP:
                mean, cov = model(x_test, full_cov=True)
            else:
                mean, cov = model(x_test, full_cov=True, noiseless=False)
        sd = cov.diag().sqrt()  # standard deviation at each input point x
        plt.plot(x_test.numpy(), mean.numpy(), 'r', lw=2)  # plot the mean
        plt.fill_between(x_test.numpy(),  # plot the two-sigma uncertainty about the mean
                         (mean - 30.0 * sd).numpy(),
                         (mean + 30.0 * sd).numpy(),
                         color='C0', alpha=0.3)
    if n_prior_samples > 0:  # plot samples from the GP prior
        if x_test is None:
            x_test = torch.linspace(-0.5, 5.5, n_test)  # test inputs
        noise = (model.noise if type(model) != gp.models.VariationalSparseGP
                 else model.likelihood.variance)
        cov = kernel.forward(x_test) + noise.expand(n_test).diag()
        samples = dist.MultivariateNormal(torch.zeros(n_test), covariance_matrix=cov) \
            .sample(sample_shape=(n_prior_samples,))
        plt.plot(x_test.numpy(), samples.numpy().T, lw=2, alpha=0.4)
    plt.xticks(np.arange(0, first_test_index, 250))
    plt.show()


def inducing_point_gamma(n, shift):
    """
        Returns a transformed X~InvGamma(10.5, 2 * n) random sample to obtain inducing points
        that favour more recent time periods.

        The transformation is given by max(n + shift - X, 0).

        :param n: The total number of points in the data
        :param shift: A parameter to ensure the final datum is able to be sampled (without it,
        this will have probability 0)
    """
    return int(max(n + shift - pyro.sample("inducing_point_i", dist.InverseGamma(torch.tensor(10.5),
                                                                      torch.tensor(2 * n))), 0))
def Gaussiansparse(x1,x2,Y1,Y2,Xu):
# Number of intervals in entire dataset
# N = df.shape[0]

# Collection of all intervals (i.e. 1-Jan-10 has intervals 1, ..., 48, 2-Jan-10 has intervals
# 49, ... 96, etc.)
 #T = range(1, N + 1)

# Test: 1 Jan 2010 - 31 Dec 2016
# Train: 1 Jan 2017 - 30 September 2021
 smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
 pyro.set_rng_seed(0)
 first_test_index = 3000#12273#6

 T_train = x1
 T_test = x2

#spot_prices = df.loc[:, ['Spot_Price']]

 spot_prices_train = Y1
#spot_prices.iloc[[t - 1 for t in T_train]].transpose().values
 spot_prices_test = Y2
#spot_prices.iloc[[t - 1 for t in T_test]]

 #X = torch.tensor(list(T_train)).float()
 X=torch.tensor(T_train).float()
# X = X.type('torch.DoubleTensor')
 Xtest=torch.tensor(T_test).float()
 #Xtest = torch.tensor(list(T_test)).float()

 y=torch.tensor(spot_prices_train).float()
 y = torch.transpose(y, 0, 1)
 #print(len(X))
 #print(len(y))
 #reshape(first_test_index,)
# y = y.type('torch.DoubleTensor')

 #Xu = torch.tensor([x1[i] for i in range(0, 480, 49)] + [x1[i] for i in range(480, first_test_index,
                                                                           #25)]).float()
#
# Xu = torch.tensor([inducing_point_gamma(first_test_index, 250) for _ in range(100)]).float()
 #print(Xu)
 pyro.clear_param_store()
 kernel = gp.kernels.RBF(input_dim=1, variance=torch.tensor(5.), lengthscale=torch.tensor(10.))

 gpr = gp.models.SparseGPRegression(X=X, y=y, kernel=kernel, Xu=Xu, jitter=1.0e-5)

# note that our priors have support on the positive reals
 gpr.kernel.lengthscale = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))
 gpr.kernel.variance = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))

 optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
 loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
 losses = []

 num_steps = 2500 if not smoke_test else 2
 for i in range(num_steps):
    optimizer.zero_grad()
    loss = loss_fn(gpr.model, gpr.guide)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
 plt.plot(losses)
 plt.show()

 plot(model=gpr, plot_observed_data=True, plot_predictions=True, x_test=Xtest, x_train=X,
     y_train=y)

 post_mean, post_cov = gpr.forward(Xtest, full_cov=False)

 R2 = r2_score(spot_prices_test.astype(float), post_mean.detach().numpy())
 print(R2)
 return (post_mean,post_cov,R2)
