import os
import matplotlib.pyplot as plt
import torch

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import math
import timeit

"""
    (Initial/Basic) GP for spot prices based on time.
    
    Author: Adrian Rajkamal
    Date: 16/12/2021
"""

smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
pyro.set_rng_seed(0)
DIR_BASE='D:/Documents/OneDrive/Documents/Data Science/MaxKelsen/csenergy-internship/data/processed'
FILE_HISTORY1=os.path.join(DIR_BASE,"historical_spot_price.csv")
df=pd.read_csv(FILE_HISTORY1)
#df = pd.read_csv("../../data/processed/historical_spot_price.csv")

""" HELPER FUNCTIONS """


def plot(plot_observed_data=False, plot_predictions=False, n_prior_samples=0, model=None,
         x_test=None, x_train=None, y_train=None):
    """ Plotting code for GP posterior - from GP Pyro tutorial"""
    n_test = len(x_test)
    plt.figure(figsize=(12, 6))
    if plot_observed_data:
        plt.plot(x_train.numpy(), y_train.numpy(), 'kx')
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


def inducing_point_gamma(n, alpha, beta, shift, num_inducing):
    """
        Returns a transformed InvGamma(alpha, beta) random sample to obtain inducing points
        that favour more recent time periods. Points are evenly generated across the training
        dataset, and then using this InvGamma transformation, more recent points are sampled.

        The transformation is given by floor(max(n + shift - X, 0)).

        :param n: The total number of points in the data
        :param alpha: concentration parameter
        :param beta: rate parameter
        :param shift: A parameter to ensure the final datum is able to be sampled (without it,
        this will have probability 0)
        :param num_inducing: Number of inducing points required
    """
    Xu = {i for i in range(0, n, math.ceil((2 * n) / num_inducing))}
    num_gamma = math.floor(num_inducing / 2)

    Xu = Xu.union({math.floor(max(n + shift - pyro.sample("inducing_point_i", dist.InverseGamma(
        torch.tensor(alpha), torch.tensor(beta))), 0)) for _ in range(num_gamma)})

    while len(Xu) != num_inducing:
        Xu.add(math.floor(max(n + shift - pyro.sample("inducing_point_i", dist.InverseGamma(
        torch.tensor(alpha), torch.tensor(beta))), 0)))

    # Solve positive definite issues by eliminating numerical linear dependence via spreading out
    # inducing points more
    Xu = sorted(list(Xu))
    Xu = [Xu[i] for i in range(0, len(Xu), 5)]
    return torch.tensor(Xu).float()
    # return math.floor(max(n + shift - pyro.sample("inducing_point_i", dist.InverseGamma(
    #     torch.tensor(alpha), torch.tensor(beta))), 0))

# Number of intervals in entire dataset
N = df.shape[0]

# Collection of all intervals (i.e. 1-Jan-10 has intervals 1, ..., 48, 2-Jan-10 has intervals
# 49, ... 96, etc.)
T = range(1, N + 1)

<<<<<<< HEAD
# Test: 1 Jan 2010 - 31 Dec 2016
# Train: 1 Jan 2017 - 30 September 2021
first_test_index = 160000#12273#6
=======
# Train: 1 Jan 2010 - 31 Dec 2016
# Test: 1 Jan 2017 - 30 September 2021
first_test_index = 4000#12273#6
>>>>>>> d6974ca9a0d9beeb011b28d3fd5465bfef8037e0

T_train = T[:first_test_index]
T_test = T[first_test_index:]

t0 = timeit.default_timer()
spot_prices = df.loc[:, ['Spot_Price']]

spot_prices_train = spot_prices.iloc[[t - 1 for t in T_train]].transpose().values
spot_prices_test = spot_prices.iloc[[t - 1 for t in T_test]]

X = torch.tensor(list(T_train)).float()
# X = X.type('torch.DoubleTensor')

Xtest = torch.tensor(list(T_test)).float()

y = torch.tensor(spot_prices_train).float().reshape(first_test_index,)
# y = y.type('torch.DoubleTensor')

num_inducing = 100
Xu = inducing_point_gamma(first_test_index, 20., 3. * first_test_index, first_test_index/22, num_inducing)
# Xu = torch.tensor([X[i] for i in range(0, 480, 49)] + [X[i] for i in range(480, first_test_index,
#                                                                            25)]).float()
# Xu = {inducing_point_gamma(first_test_index, 10.5, 8.75 * first_test_index, 1050, 100)
#       for _ in range(num_inducing)}

# while len(Xu) != num_inducing:
#     Xu.add(inducing_point_gamma(first_test_index, 10.5, 8.75 * first_test_index, 1050, 100))

# Solve positive definite issues by eliminating numerical linear dependence via spreading out
# inducing points more
# Xu = sorted(list(Xu))
# Xu = [Xu[i] for i in range(0, len(Xu), 5)]
# Xu = torch.tensor(Xu).float()

pyro.clear_param_store()
kernel = gp.kernels.RBF(input_dim=1, variance=torch.tensor(5.), lengthscale=torch.tensor(10.))

gpr = gp.models.SparseGPRegression(X, y, kernel, Xu=Xu, jitter=1.0e-5)

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
print("Elapsed Time:", timeit.default_timer() - t0, "seconds")
plt.plot(losses)
plt.show()

plot(model=gpr, plot_observed_data=True, plot_predictions=True, x_test=Xtest, x_train=X,
     y_train=y)

post_mean, post_cov = gpr.forward(Xtest, full_cov=False)

R2 = r2_score(spot_prices_test.astype(float), post_mean.detach().numpy())
print(R2)
