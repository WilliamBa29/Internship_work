import os
import matplotlib.pyplot as plt
import torch

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import math
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error
import math
import timeit
import numpy as np
from src.modelling.Gaussian_Process2 import splitdata2,GPenergy2
from src.modelling.basic_sparse_GP1 import inducing_point_gamma,Gaussiansparse1,plot
from src.helpers.splitdata3 import splitdata3
def testing2(x1,x2,Y1,Y2,notestpoints,df_,scaler1,scaler,num_inducing,kernel,mean_function,I,d,t1,t2,X,X1):
    
    
    """Function used to forecast multiple steps ahead with regard to the GP model. The model is initially trained on the training data, x1, Y1 and then predicts
    notestpoints ahead, then the posterior model is updated with these points (but not optimized), and the next notestpoints ahead are predicted. This process continues until the
     entire validation/testdata set (x2,Y2) has been predicted

    Args:
        x1 ([numpy array]): [An array where each column represents the values of one predictor variable for each sample in the training data]
        x2 ([type]): [An array where each column represents the values of one predictor variable for each sample in the training data]

        Y1 ([type]): [A vector consisting of the values of the response/target variable for each sample in the training data]
        Y2 ([type]): [A vector consisting of the values of the response/target variable for each sample in the testing datdescription]
        notestpoints ([type]): [The number of intervals/samples ahead that one wishes to predict]
        df_ ([type]): [Original CapPrice dataset]
        scaler1 ([type]): [The scaler used to normalize/standardize the target variable vectors in the function, 'splitdata3.py']
        scaler ([type]): [The scaler used to normalize/standardize the predictor variable vectors in the function, 'splitdata3.py']
        Returns:
        The relative absolute error of the produced predictions on the validation data, where in the relative absolute mean the mean 'y/target' value
         is relative to the half hour interval in question 

    """
    post_mean=np.empty([notestpoints,math.ceil(len(x2)/notestpoints)],dtype='object')
    post_cov1=np.empty([notestpoints,math.ceil(len(x2)/notestpoints)],dtype='object')
    #print(kernel)
    
    #Xu=inducing_point_gamma(n=len(x1)+(2)*notestpoints,alpha=14.,beta=float(1.5625*(len(x1)+(2)*notestpoints)),shift=(len(x1)+(2)*notestpoints)/22, num_inducing=num_inducing,x1=np.concatenate((x1,x2[range(0,(2)*notestpoints),:])))
    #gpr = gp.models.SparseGPRegression(X=torch.tensor(np.concatenate((x1,x2[range(0,(2)*notestpoints),:]),axis=0).astype(float)), y=torch.transpose(torch.tensor(np.concatenate((Y1,Y2[range(0,(2)*notestpoints),:]),axis=0).astype(float)),0,1), kernel=kernel, mean_function=mean_function,Xu=Xu, jitter=1.0e-5)             
    
        #gpr=Gaussiansparse1(x1=np.concatenate((x1,x2[range(0,(i-1)*notestpoints),:]),axis=0),Y1=np.concatenate((Y1,Y2[range(0,(i-1)*notestpoints),:]),axis=0),Xu=Xu,kernel=kernel,mean_function=mean_function,I=I)
    for i in range(1,math.ceil(len(x2)/notestpoints)+1):
        #print(len(np.concatenate((Y1,Y2[range(0,(i-1)*notestpoints),:]),axis=0)))
        #print(np.shape(np.concatenate((x1,x2[range(0,(i-1)*notestpoints),:]),axis=0).reshape(1,-1)))
        #print(len(x1))
        #print(x2[range(0,(i-1)*notestpoints),:])
       # print(len(np.transpose(np.concatenate((x1,x2[range(0,(i-1)*notestpoints),:]),axis=0).reshape(1,-1))))
        if i==1:
         #print('hello')
         #print(len(x1))
         if I==3:
             kernel1=gp.kernels.RBF(input_dim=1)
             kernel2=gp.kernels.Periodic(input_dim=1)
             kernel= gp.kernels.Sum(kernel1,kernel2)
         if I==4:
             kernel1=gp.kernels.Periodic(input_dim=1)
             kernel2=gp.kernels.Periodic(input_dim=1)
             kernel= gp.kernels.Sum(kernel1,kernel2)
         if I==5:
             kernel1=gp.kernels.RBF(input_dim=1)
             kernel2=gp.kernels.Periodic(input_dim=1)  
             kernel3=gp.kernels.Linear(input_dim=1)
             kernela=gp.kernels.Sum(kernel1,kernel2)
             kernel=gp.kernels.Sum(kernela,kernel3) 
         if I==7:
             kernel1=gp.kernels.RBF(input_dim=1)
             kernel2=gp.kernels.Periodic(input_dim=1)
             kernel= gp.kernels.Product(kernel1,kernel2)    

         #print(kernel)
         Xu=inducing_point_gamma(n=len(x1)+(i-1)*notestpoints,alpha=14.,beta=float(1.5625*(len(x1)+(i-1)*notestpoints)),shift=(len(x1)+(i-1)*notestpoints)/22, num_inducing=num_inducing,x1=np.concatenate((x1,x2[range(0,(i-1)*notestpoints)])))
         print(np.shape(np.concatenate((x1,x2[range(0,(i-1)*notestpoints)]),axis=0).reshape(-1,1)))
         gpr=Gaussiansparse1(x1=np.concatenate((x1,x2[range(0,(i-1)*notestpoints)]),axis=0).reshape(-1,1),Y1=np.concatenate((Y1,Y2[range(0,(i-1)*notestpoints),:]),axis=0),Xu=Xu,kernel=kernel,mean_function=mean_function,I=I)
        else:
         #kernel = gp.kernels.RBF(input_dim=1, variance=torch.tensor(gpr.kernel.variance.item()), lengthscale=torch.tensor(gpr.kernel.lengthscale.item()))
         #print(gpr.kernel)
         #if I==3:
           #  kernel1=gp.kernels.RBF(input_dim=1,variance=torch.tensor(gpr.kernel.kern0.variance.item()),lengthscale=torch.tensor(gpr.kernel.kern0.lengthscale.item()))
           #  kernel2=gp.kernels.Periodic(input_dim=1,variance=torch.tensor(gpr.kernel.kern1.variance.item()),lengthscale=torch.tensor(gpr.kernel.kern1.lengthscale.item()))
          #   kernel= gp.kernels.Sum(kernel1,kernel2)
         #if I==4:
            # kernel1=gp.kernels.Periodic(input_dim=1,variance=torch.tensor(gpr.kernel.kern0.variance.item()),lengthscale=torch.tensor(gpr.kernel.kern0.lengthscale.item()))
           #  kernel2=gp.kernels.Periodic(input_dim=1,variance=torch.tensor(gpr.kernel.kern1.variance.item()),lengthscale=torch.tensor(gpr.kernel.kern1.lengthscale.item()))
          #   kernel= gp.kernels.Sum(kernel1,kernel2)    
         #if I==5:
           # kernel1=gp.kernels.RBF(input_dim=1,variance=torch.tensor(gpr.kernel.kern0.kern0.variance.item()),lengthscale=torch.tensor(gpr.kernel.kern0.kern0.lengthscale.item()))
           # kernel2=gp.kernels.Periodic(input_dim=1,variance=torch.tensor(gpr.kernel.kern0.kern1.variance.item()),lengthscale=torch.tensor(gpr.kernel.kern0.kern1.lengthscale.item()))
            # kernela= gp.kernels.Sum(kernel1,kernel2)
          #  kernel=gp.kernels.Sum(kernela,kernel3) 
         #if I==7:
            #kernel1=gp.kernels.RBF(input_dim=1,variance=torch.tensor(gpr.kernel.kern0.variance.item()),lengthscale=torch.tensor(gpr.kernel.kern0.lengthscale.item()))
            #kernel2=gp.kernels.Periodic(input_dim=1,variance=torch.tensor(gpr.kernel.kern1.variance.item()),lengthscale=torch.tensor(gpr.kernel.kern1.lengthscale.item()))
            #kernel= gp.kernels.Product(kernel1,kernel2)        
            #print(torch.tensor(np.concatenate((x1,x2[range(0,(i-1)*notestpoints)]),axis=0).astype(float).reshape(1,-1)).shape)
            #print(torch.transpose(torch.tensor(np.concatenate((Y1,Y2[range(0,(i-1)*notestpoints),:]),axis=0).astype(float)),0,1))
         gpr.set_data(torch.transpose(torch.tensor(np.concatenate((x1,x2[range(0,(i-1)*notestpoints)]),axis=0).astype(float).reshape(1,-1)),0,1), y=torch.transpose(torch.tensor(np.concatenate((Y1,Y2[range(0,(i-1)*notestpoints),:]),axis=0).astype(float)),0,1))
         #gpr = gp.models.SparseGPRegression(X=torch.transpose(torch.tensor(np.concatenate((x1,x2[range(0,(i-1)*notestpoints)]),axis=0).astype(float).reshape(1,-1)),0,1), y=torch.transpose(torch.tensor(np.concatenate((Y1,Y2[range(0,(i-1)*notestpoints),:]),axis=0).astype(float)),0,1), kernel=kernel, mean_function=mean_function,Xu=gpr.Xu.data, jitter=1.0e-5)            
        for j in range(1,min(notestpoints+1,len(x2)-(i-1)*notestpoints+1)):
            #print(j)
           
             
         print(np.shape(Y1))
         print(np.shape(x2[(i-1)*notestpoints+j-1].numpy().reshape(1,-1)))
             #rint(np.shape(x2[(i-1)*notestpoints+j-1,:]))
         c, post_cov = gpr.forward(torch.tensor(x2[(i-1)*notestpoints+j-1].numpy().reshape(1,-1).astype(float)), full_cov=False)
         post_mean[j-1,i-1]=c.detach().numpy()
         post_cov1[j-1,i-1]=post_cov.detach().numpy()
            
    M=post_mean.flatten()
    M=M[M!=None]
    C=post_cov1.flatten()
    C=C[C!=None]
    print(d)
    #rae=RAE(Y2,Y2,x2.astype(float),x2.astype(float),M,scaler1,scaler,d)
    rae1=RAE(Y2,Y1,x2,x1,M,scaler1,scaler,d,X,X1)
    Msq=mean_squared_error(Y2,M)
    #print(t1.reshape(1,-1).numpy())
    plot(plot_observed_data=True, plot_predictions=True, n_prior_samples=0, model=None,
         x_test=t2, x_train=t1, y_train=Y1,y_test=Y2,mean=M,cov=C,scaler1=scaler1,d=d)
    
    #print(rae)
    R2 = r2_score(Y2.astype(float), np.transpose(M))
    print(R2)
    return(Msq,M,C,rae1)#print(R2)

def RAE(Y2,Y1,x2,x1,posterior_mean,scaler1,scaler,d,X,X1):
    """[A function which takes in the training and validation data as well as a prediction over the validation data (posterior mean)
    and produces the RAE of the prediction over the validation dataset where the mean value used in the RAE is specific to a half hour interval]

    Args:
        Y2 ([array/vector]): [The column containing the target values of the validation data set]
        Y1 ([array/vector]): [The column containing the target values of the training data set]
        x2 ([array): [An array where each column contains the values of a predictor variable for each sample in the validation dataset]
        x1 ([array]): [An array where each column contains the values of a predictor variable for each sample in the training datasetdescription]
        posterior_mean ([array/vector]): [A vector containing the mean of the posterior over the validation dataset]
        scaler1 ([]): [The scaler used on the vectors of target values (both training and validation)]
        scaler ([type]): [The scaler used on the array of predictor variable values (both training and validation)]

        Returns: Returns the relative absolute errror of the posterior mean relative to the validation data set where the 
        mean value used in the RAE is specific to a half hour interval.   
    
    """
    print(d)
    #print(d!='none')
    if d!='none':
     Y1=scaler1.inverse_transform(Y1)
     Y2=scaler1.inverse_transform(Y2)
     posterior_mean=scaler1.inverse_transform(posterior_mean.reshape(-1,1))
    #Comparison=x1==x2
    #print(Comparison)
    #print(Comparison.all())
    #print(np.array_equal(x1,x2,equal_nan=True))
    
     

    sum2=0
    sum1=0
    interval_mean=np.empty(48,dtype='object')
    #print(len(posterior_mean))
    #print(len(Y2))
    #print(x1[range(0,48),0])
    #print(x1[range(0,48),0].astype(int)==15)
    for z in range(0,48):
        interval_mean[z]=np.mean(Y1[X[:,0].astype(int)==z+1])
    for i in range(0,len(Y2)):
       # print(posterior_mean[i])
        sum2=sum2+abs(Y2[i]-posterior_mean[i])
    for j in range(0, len(Y2)):
       # print(Y2[j])
        sum1=sum1+ abs(Y2[j]-interval_mean[int(X1[j,0]-1)])
    #print(sum1)
    #print(sum)
    #print(interval_mean)
   # print(sum((x1[:,0]==15).astype(int)))
    
    RAE=sum2/sum1
    print(posterior_mean)
    print(interval_mean)
    print(sum2)
    print(sum1)
    return(RAE)    


        