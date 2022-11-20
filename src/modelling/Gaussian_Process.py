import csv
import numpy as np
from numpy.core.defchararray import lower
import pandas as pd
import datetime
from multiprocessing import Pool
from multiprocessing import sharedctypes
import os
import sys
import torch
import scipy
from scipy import stats
import math
import sklearn
from sklearn.metrics import r2_score


from itertools import repeat

import itertools

sys.path.insert(0,'D:\Documents\OneDrive\Documents\Data Science\MaxKelsen\csenergy-internship')
from src.helpers.slicingdata import formatting
from src.helpers.slicingdata import week
from src.helpers.slicingdata1 import rearrange
from src.helpers.frequency import relfreq
import matplotlib.pyplot as plt
import torch

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
def splitdata(df_,month,work):
 m=df_.iloc[:,0].str[3:6]
 arr = np.empty(len(df_), dtype=object)
 arr[m=='Jan']=1
 arr[m=='Feb']=2
 arr[m=='Mar']=3
 arr[m=='Apr']=4
 arr[m=='May']=5
 arr[m=='Jun']=6
 arr[m=='Jul']=7
 arr[m=='Aug']=8
 arr[m=='Sep']=9
 arr[m=='Oct']=10
 arr[m=='Nov']=11
 arr[m=='Dec']=12
 b,c=df_,arr
 

 c2=np.zeros(len(df_))
 if type(month)==str:
    
   f=(month==df_.iloc[:,0].str[3:6])
   f=np.array(f)
   f=f.astype(int)
   c2=c2+f
 else:
  for r in range(0,len(month)):
     
     f=(month[r]==df_.iloc[:,0].str[3:6])
     
     f=np.array(f)
     f=f.astype(int)
     c2=c2+f
 
 c2=(c2==1)
 M2=[i for i, x in enumerate(c2) if x]
 P=Pool(os.cpu_count()-1)
 arguments=[(a,b,c) for a in M2]
 E=np.zeros(len(df_))
 E1=np.array(P.starmap(week,arguments))
 E[M2]=E1
 
 if work=='Workdays':
  data=df_.iloc[(E==1),2]
 if work=='Non Workdays':
  data=df_.iloc[(E==0),2]
 return data
#DIR_BASE = 'D:/documents'
#FILE_HISTORY = os.path.join(DIR_BASE, "CapPrice.csv")
#df_=pd.read_csv(FILE_HISTORY)
def GPenergy(data,validationdata,kernel):
    #s=np.empty(len(df_),dtype=object)
 

 x=np.empty(len(data),dtype=object)
 x1=np.empty(len(validationdata),dtype=object)
 for i in range(1,len(data)+1):
  x[i-1]=i
 for j in range(len(data)+1,len(data)+1+len(validationdata)):
  x1[j-len(data)-1]=j
  #print(x1[j-len(data)-1])
 #print(x1)
 #print(type(x1))
 xj=torch.tensor(list(x))
 x1j=torch.tensor(list(x1))  
 Y=torch.tensor(list(data))

 Y1=torch.tensor(list(validationdata))
 #kernel = gp.kernels.RBF(input_dim=1, variance=torch.tensor(5.),
                        #lengthscaltorch.tensor(10.))
 gpr = gp.models.GPRegression(xj, Y, kernel,mean_function=lambda x: 0.2, noise=torch.tensor(1.))     
 print(gpr.parameters())
 optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
 loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
 losses = []
 num_steps = 3# if not smoke_test else 2
 for i in range(num_steps):
    print(i)
    optimizer.zero_grad()
    loss = loss_fn(gpr.model, gpr.guide)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
 #kernel = gp.kernels.RBF(input_dim=1)

 print(gpr.kernel)
 #print(gpr.kernel.lengthscale)
 #print(gpr.kernel.period)
 #print(gpr.noise)
#gpr.noise
 mean1, cov=gpr.forward(x1j, full_cov=False, noiseless=False)
 p_value=np.empty(len(validationdata),dtype=object)
 lower=np.empty(len(validationdata),dtype=object)
 upper=np.empty(len(validationdata),dtype=object)
 Rsq=sklearn.metrics.r2_score(validationdata,mean1.detach().numpy())
 lower=mean1.detach().numpy()-1.96*np.sqrt(cov.detach().numpy())
 upper=mean1.detach().numpy()+1.96*np.sqrt(cov.detach().numpy())
#gpr = gp.models.GPRegression(X, y, kernel)
#GPregression(x,df_.iloc[2,:])
 #Xnew = torch.tensor([[2., 3, 1]])
 #gpr.forward(x_new)
 #print(len(x1))
 for i in range(0,len(x1)):
 # print(mean1[i].detach().numpy())
#  print(cov[i].detach().numpy())
  p_value[i]=1-scipy.stats.norm.cdf(300,loc=mean1[i].detach().numpy(),scale=cov[i].detach().numpy())
 import matplotlib.pyplot as plt
 plt.plot(x1,mean1.detach().numpy(), label='Mean (Spot price) of the GP Posterior')
 plt.plot(x1,validationdata,label='Spot Price')
 plt.plot(x1,lower,label='Lower Bound of Posterior Spot Price 95% Predictive Interval')
 plt.plot(x1,upper,label='Upper Bound of Posterior Spot Price 95% Predictive Interval')
 plt.legend()
 plt.show()
 #plt.plot(model=gpr, plot_observed_data=True, plot_predictions=True)
 #plt.show()
 return lower,upper,mean1,cov,p_value,Rsq
#f_loc, f_cov = gpr(Xnew, full_cov=True)