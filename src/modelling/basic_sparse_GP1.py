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

#df = pd.read_csv("../../data/processed/historical_spot_price.csv")

""" HELPER FUNCTIONS """
def plot1(x_test=None, x_val=None, y_val=None,y_test=None, mean=None,cov=None,scaler1=None,d=None,meanval=None,covval=None):
         
    """ Plotting code for GP posterior - from GP Pyro tutorial"""
    n_test = len(x_test)
    fig, axs = plt.subplots(2)
    fig.suptitle('Validation and Test Predictions, January 2010 - 2021')
    #plt.figure(figsize=(12, 6),constrained_layout=True)
    
   
        
    
    
    sdval=np.sqrt(covval.astype(float))
    sd=np.sqrt(cov.astype(float))
    if d!='none':
          #plt.plot(x_test.detach().numpy(),np.reshape(scaler1.inverse_transform(mean),-1), 'r', lw=2,marker='o')  # plot the mean
          axs[0].plot(x_val.reshape(1,-1).numpy(),scaler1.inverse_transform(np.transpose(y_val)), label='Raw')
          axs[0].plot(x_val.detach().numpy(),np.reshape(scaler1.inverse_transform(meanval.reshape(1,-1)),-1),label='Validation')
          axs[0].fill_between(x_val.numpy(),  # plot the two-sigma uncertainty about the mean
                         np.reshape(scaler1.inverse_transform((meanval - 30 * sdval).reshape(1,-1)),-1).astype(float),
                         np.reshape(scaler1.inverse_transform((meanval + 30 * sdval).reshape(1,-1)),-1).astype(float),
                         color='C0', alpha=0.3,label='Interval' )
          axs[1].plot(x_test.detach().numpy(),np.reshape(scaler1.inverse_transform(mean.reshape(1,-1)),-1),label='Validation')
          axs[1].fill_between(x_test.numpy(),  # plot the two-sigma uncertainty about the mean
                         np.reshape(scaler1.inverse_transform((mean - 30 * sd).reshape(1,-1)),-1).astype(float),
                         np.reshape(scaler1.inverse_transform((mean + 30 * sd).reshape(1,-1)),-1).astype(float),
                         color='C0', alpha=0.3,label='Interval' ) 
          axs[1].plot(x_test.reshape(1,-1).numpy(),scaler1.inverse_transform(np.transpose(y_test)),'kx',label='Raw')                             
    else:
          axs[0].plot(x_val.detach().numpy(),np.reshape(meanval,-1), 'r', lw=2,marker='o',label='Validation') 
          axs[0].plt.fill_between(x_val.numpy(),  # plot the two-sigma uncertainty about the mean
                         np.reshape((meanval - 30 * sdval),-1).astype(float),
                         np.reshape((meanval + 30 * sdval),-1).astype(float),
                         color='C0', alpha=0.3,label='Interval') 
          axs[0].plot(x_val.reshape(1,-1).numpy(),np.transpose(y_val), label='Raw')
          axs[1].plot(x_test.detach().numpy(),np.reshape(meanval,-1), 'r', lw=2,marker='o',label='Validation') 
          axs[1].fill_between(x_test.numpy(),  # plot the two-sigma uncertainty about the mean
                         np.reshape((mean - 30 * sd),-1).astype(float),
                         np.reshape((mean + 30 * sd),-1).astype(float),
                         color='C0', alpha=0.3,label='Interval') 
          axs[1].plot(x_test.reshape(1,-1).numpy(),np.transpose(y_test),'kx',label='Raw')               
          #axs[0].plot(x_test.detach().numpy(),y_val,label='Test Spot Prices') 
     # if d!='none':
         
           #label='Predictive Interval for Test Spot Price Predictions'              
        
        #kernel = gp.kernels.RBF(input_dim=1, variance=torch.tensor(5.), lengthscale=torch.tensor(10.))
        #cov = kernel.forward(x_test) + noise.expand(n_test).diag()
        #samples = dist.MultivariateNormal(torch.zeros(n_test), covariance_matrix=cov) \
         #   .sample(sample_shape=(n_prior_samples,))
        #plt.plot(x_test.numpy(), samples.numpy().T, lw=2, alpha=0.4)
    plt.xlabel('Interval Number ')
    plt.ylabel('Spot Price')
    plt.xlabel('Interval Number ')
    plt.ylabel('Spot Price')
    #plt.xticks(np.arange(0, first_test_index, 250))
    axs[0].legend()
    axs[1].legend()
    plt.tight_layout()
    plt.show()

def plot(plot_observed_data=False, plot_predictions=False, n_prior_samples=0, model=None,
         x_test=None, x_train=None, y_train=None,y_test=None, mean=None,cov=None,scaler1=None,d=None):
    """ Plotting code for GP posterior - from GP Pyro tutorial"""
    n_test = len(x_test)
    plt.figure(figsize=(12, 6),constrained_layout=True)
    if plot_observed_data:
        m= x_train.reshape(1,-1).numpy()
        if d!='none':
         
         c=scaler1.inverse_transform(np.transpose( y_train))
        else:
         c=np.transpose( y_train)    
        plt.plot(m,c, 'kx',label='Training & Validation Spot Prices')
        plt.xlabel('Interval Number ')
        plt.ylabel('Spot Price')
    if plot_predictions:
        if x_test is None:
            x_test = torch.linspace(-0.5, 5.5, n_test)  # test inputs
        # compute predictive mean and variance
        #with torch.no_grad():
         #   if type(model) == gp.models.VariationalSparseGP:
          #      mean, cov = model(x_test, full_cov=True)
           # else:
            #    mean, cov = model(x_test, full_cov=True, noiseless=False)
        #print('hello')
        #print(mean.detach().numpy())
        #print(x_test.detach().numpy().reshape(1,-1))
        sd=np.sqrt(cov.astype(float))
        #sd = cov.diag().sqrt()  # standard deviation at each input point x
        print(np.shape(y_test))
        print(np.shape(mean))
        print(np.shape(x_test.detach().numpy()))
        #print(np.shape(scaler1.inverse_transform(mean)))
        if d!='none':
          #plt.plot(x_test.detach().numpy(),np.reshape(scaler1.inverse_transform(mean),-1), 'r', lw=2,marker='o')  # plot the mean
          plt.plot(x_test.detach().numpy(),np.reshape(scaler1.inverse_transform(mean.reshape(1,-1)),-1),label='Test Spot Price Predictions')
          plt.plot(x_test.detach().numpy(),np.reshape(scaler1.inverse_transform(y_test),-1),label='Test Spot Prices')
        else:
          plt.plot(x_test.detach().numpy(),np.reshape(mean,-1), 'r', lw=2,marker='o',label='Test Spot Price Predictions')  
          plt.plot(x_test.detach().numpy(),y_test,label='Test Spot Prices')         
        
        if d!='none':
         plt.fill_between(x_test.numpy(),  # plot the two-sigma uncertainty about the mean
                         np.reshape(scaler1.inverse_transform((mean - 30 * sd).reshape(1,-1)),-1).astype(float),
                         np.reshape(scaler1.inverse_transform((mean + 30 * sd).reshape(1,-1)),-1).astype(float),
                         color='C0', alpha=0.3,label='Predictive Interval for Test Spot Price Predictions' )
           #label='Predictive Interval for Test Spot Price Predictions'              
        else:
         plt.fill_between(x_test.numpy(),  # plot the two-sigma uncertainty about the mean
                         np.reshape((mean - 30 * sd),-1).astype(float),
                         np.reshape((mean + 30 * sd),-1).astype(float),
                         color='C0', alpha=0.3,label='Predictive Interval for Test Spot Price Predictions')
                          
    if n_prior_samples > 0:  # plot samples from the GP prior
        if x_test is None:
            x_test = torch.linspace(-0.5, 5.5, n_test)  # test inputs
        noise = (model.noise if type(model) != gp.models.VariationalSparseGP
                 else model.likelihood.variance)
        #kernel = gp.kernels.RBF(input_dim=1, variance=torch.tensor(5.), lengthscale=torch.tensor(10.))
        #cov = kernel.forward(x_test) + noise.expand(n_test).diag()
        #samples = dist.MultivariateNormal(torch.zeros(n_test), covariance_matrix=cov) \
         #   .sample(sample_shape=(n_prior_samples,))
        #plt.plot(x_test.numpy(), samples.numpy().T, lw=2, alpha=0.4)
    #plt.xticks(np.arange(0, first_test_index, 250))
    #plt.legend(loc='lower right')
    plt.show()

def inducing_point_gamma(n, alpha, beta, shift, num_inducing,x1):
    """
        Returns a transformed InvGamma(alpha, beta) random sample to obtain inducing points
        that favour more recent time periods.

        The transformation is given by floor(max(n + shift - X, 0)).

        :param n: The total number of points in the data
        :param alpha: concentration parameter
        :param beta: rate parameter
        :param shift: A parameter to ensure the final datum is able to be sampled (without it,
        this will have probability 0)
        :param num_inducing: Number of inducing points required
    
    """
    #print(n)
    #print(alpha)
    #print(beta)
    Xu = {math.floor(max(n + shift - pyro.sample("inducing_point_i", dist.InverseGamma(
        torch.tensor(alpha), torch.tensor(beta))), 0)) for _ in range(num_inducing)}
    #print('hello2')
    while len(Xu) != num_inducing:
      #  print(Xu)
        #print(n + shift - pyro.sample("inducing_point_i", dist.InverseGamma(
        #torch.tensor(alpha), torch.tensor(beta))))
        Xu.add(math.floor(max(n + shift - pyro.sample("inducing_point_i", dist.InverseGamma(
        torch.tensor(alpha), torch.tensor(beta))), 0)))

    # Solve positive definite issues by eliminating numerical linear dependence via spreading out
    # inducing points more
    
    Xu = sorted(list(Xu))
   
    Xu = [Xu[i] for i in range(0, len(Xu), 5)]
    return torch.tensor(x1[Xu].astype(np.float))



# Number of intervals in entire dataset
#N = df.shape[0]

# Collection of all intervals (i.e. 1-Jan-10 has intervals 1, ..., 48, 2-Jan-10 has intervals
# 49, ... 96, etc.)
#T = range(1, N + 1)

# Test: 1 Jan 2010 - 31 Dec 2016
# Train: 1 Jan 2017 - 30 September 2021
#first_test_index = 4000#12273#6

#T_train = T[:first_test_index]
#T_test = T[first_test_index:]

#t0 = timeit.default_timer()
#spot_prices = df.loc[:, ['Spot_Price']]

#spot_prices_train = spot_prices.iloc[[t - 1 for t in T_train]].transpose().values
#spot_prices_test = spot_prices.iloc[[t - 1 for t in T_test]]

#X = torch.tensor(list(T_train)).float()
# X = X.type('torch.DoubleTensor')

#Xtest = torch.tensor(list(T_test)).float()[:500]

#y = torch.tensor(spot_prices_train).float().reshape(first_test_index,)
# y = y.type('torch.DoubleTensor')

#num_inducing = 100
#Xu = inducing_point_gamma(first_test_index, 10.5, 8.75 * first_test_index, 1050, num_inducing)
# Xu = torch.tensor([X[i] for i in range(0, 480, 49)] + [X[i] for i in range(480, first_test_index,
#                                                                            25)]).float()
# Xu = {inducing_point_gamma(first_test_index, 10.5, 8.75 * first_test_index, 1050)
#       for _ in range(num_inducing)}

# while len(Xu) != num_inducing:
#     Xu.add(inducing_point_gamma(first_test_index, 10.5, 8.75 * first_test_index, 1050))

# Solve positive definite issues by eliminating numerical linear dependence via spreading out
# inducing points more
# Xu = sorted(list(Xu))
# Xu = [Xu[i] for i in range(0, len(Xu), 5)]
# Xu = torch.tensor(Xu).float()

#pyro.clear_param_store()
#kernel = gp.kernels.RBF(input_dim=1, variance=torch.tensor(10000.), lengthscale=torch.tensor(10.))

#gpr = gp.models.SparseGPRegression(X, y, kernel, Xu=Xu, jitter=1.0e-5)

# note that our priors have support on the positive reals
#gpr.kernel.lengthscale = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))
#gpr.kernel.variance = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))

#optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
#loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
#losses = []

#num_steps = 2500 if not smoke_test else 2
#for i in range(num_steps):
 #   optimizer.zero_grad()
 #   loss = loss_fn(gpr.model, gpr.guide)
 #   loss.backward()
 #   optimizer.step()
 #   losses.append(loss.item())
#print("Elapsed Time:", timeit.default_timer() - t0)
#plt.plot(losses)
#plt.show()

#plot(model=gpr, plot_observed_data=True, plot_predictions=True, x_test=Xtest)

#post_mean, post_cov = gpr.forward(Xtest, full_cov=False)

#R2 = r2_score(spot_prices_test.astype(float)[:500], post_mean.detach().numpy())
#print(R2)
def Gaussiansparse(x1,x2,Y1,Y2,t1,t2,Xu,scaler,scaler1):
 """
   Function which constructs and optimizes a Sparse GP regression model and then deploys it on validation data.

   Args:

   x1(array): [Array of predictor variables where each column represents the values of one predictor variable for each sample in the training data]

   x2(array): [Array of predictor variables where each column represents the values of one predictor variable for each sample in the validation data]

   Y1(array/vector): [vector of target values corresponding to the training data]

   Y2(array/vector): [vector of target values corresponding to the test data] 

   t1(arrray/vector): [vector containing the temporal (half hour) interval numbers for the samples from the training data ordered from the first interval
   in the training data to the last interval in the training data]

   t2(array/vector):[vector containing the temporal (half hour) interval numbers for the samples from the training data ordered from the first interval
   in the training data to the last interval in the validation data]

   Xu(array/vector):[A vector consisting of the inducing data point samples]

   scaler: The scaler used in 'Splitdata 3' on the array of validation and training prediction variable data

   scaler1: The scaler used in 'Splitdata3' on the array of validation and training target variable data

  Returns: The mean and covariance matrices of the posterior over the specified validation data.
  It also indicates the R-squared corresponding to the mean of the posterior over the validation data

 """
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
 X=torch.tensor(T_train.astype(float))
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

 gpr = gp.models.SparseGPRegression(X=X, y=y, kernel=kernel, mean_function=lambda x:1500, Xu=Xu, jitter=1.0e-5)

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
 plt.xlabel('No of Optimization Steps')
 plt.ylabel('Loss Function Value')
 plt.show()


 post_mean, post_cov = gpr.forward(Xtest, full_cov=False)
 y_train_unnormalized=torch.tensor(scaler1.inverse_transform(Y1)).float()
 y_test_unnormalized=torch.tensor(scaler1.inverse_transform(Y2)).float()
 #post_mean=torch.tensor(scaler1.inverse_transform(post_mean.detach().numpy()))
 plot(model=gpr, plot_observed_data=True, plot_predictions=True, x_test=t2, x_train=t1,
     y_train=y_train_unnormalized,mean=post_mean, cov=post_cov,scaler1=scaler1)

 

 R2 = r2_score(spot_prices_test.astype(float), np.transpose(post_mean.detach().numpy()))
 print(R2)
 return (post_mean,post_cov,R2)

def Gaussiansparse1(x1,Y1,Xu,kernel,mean_function,I):
 """
  Function which constructs and optimizes a Sparse GP regression model and then deploys it on validation data.

  Args:

  x1(array): [Array of predictor variables where each column represents the values of one predictor variable for each sample in the training data]

  x2(array): [Array of predictor variables where each column represents the values of one predictor variable for each sample in the validation data]

  Y1(array/vector): [vector of target values corresponding to the training data]

  Y2(array/vector): [vector of target values corresponding to the test data]

  Xu(array/vector):[A vector consisting of the inducing data point samples]

  scaler: The scaler used in 'Splitdata 3' on the array of validation and training prediction variable data

  scaler1: The scaler used in 'Splitdata3' on the array of validation and training target variable data

  Returns: The mean and covariance matrices of the posterior over the specified validation data.
  It also indicates the R-squared corresponding to the mean of the posterior over the validation data

 """    
 import pyro
 import pyro.contrib.gp as gp
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
 #print(x1)
 T_train = x1
 #T_test = x2

#spot_prices = df.loc[:, ['Spot_Price']]

 spot_prices_train = Y1
#spot_prices.iloc[[t - 1 for t in T_train]].transpose().values
 #spot_prices_test = Y2
#spot_prices.iloc[[t - 1 for t in T_test]]

 #X = torch.tensor(list(T_train)).float()
 #X=torch.tensor(T_train.astype(np.float))
 X=torch.tensor(T_train.astype(float))
# X = X.type('torch.DoubleTensor')
# Xtest=torch.tensor(T_test).float()
 #Xtest = torch.tensor(list(T_test)).float()

 y=torch.tensor(spot_prices_train.astype(float))
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
 #kernel = gp.kernels.RBF(input_dim=1, variance=torch.tensor(5.), lengthscale=torch.tensor(10.))

 gpr = gp.models.SparseGPRegression(X=X, y=y, kernel=kernel,mean_function=mean_function, Xu=Xu, jitter=1.0e-5)

# note that our priors have support on the positive reals
 if I==1 or I==2:
  gpr.kernel.lengthscale = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))
  gpr.kernel.variance = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))
 if I==3 or I==4 or I==7:
  gpr.kernel.kern0.lengthscale = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))
  gpr.kernel.kern1.lengthscale = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))
  gpr.kernel.kern0.variance = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0)) 
  gpr.kernel.kern1.variance = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))
 if I==5:
  gpr.kernel.kern0.kern0.lengthscale = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))
  gpr.kernel.kern0.kern1.lengthscale = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))
  gpr.kernel.kern1.variance= pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))
  gpr.kernel.kern0.kern0.variance = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0)) 
  gpr.kernel.kern0.kern1.variance = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0)) 

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
 #plt.plot(losses)
 #plt.show()


 
 return (gpr)

    