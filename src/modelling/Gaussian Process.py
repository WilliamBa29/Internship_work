import csv
import numpy as np
import pandas as pd
import datetime
from multiprocessing import Pool
from multiprocessing import sharedctypes
import os
import sys
import torch
from scipy import stats


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
 M2=[i for i, x in enumerate(c3) if x]
 arguments=[(a,b,c) for a in M2]
 E=np.zeros(len(df_))
 E1=np.array(P.starmap(week,arguments))
 E[M2]=E1
 
 if work=='Workdays':
  data=df_.iloc[(E==1),2]
 if work='Non Workdays':
  data=df_.iloc[(E==0),2]
 return data
#DIR_BASE = 'D:/documents'
#FILE_HISTORY = os.path.join(DIR_BASE, "CapPrice.csv")
#df_=pd.read_csv(FILE_HISTORY)
def GPenergy(data,kernel,mean,x_new):
    #s=np.empty(len(df_),dtype=object)
 

 x=np.empty(len(data),dtype=object))
 for i in range(1,len(data)+1):
        x[i]=i
 x=torch.tensor(list(x))  
 Y=torch.tensor(list(Data.iloc[2,:]))
 kernel = gp.kernels.RBF(input_dim=1, variance=torch.tensor(5.),
                        lengthscale=torch.tensor(10.))
gpr = gp.models.GPRegression(x, y=data, kernel, noise=torch.tensor(1.))     
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
kernel = gp.kernels.RBF(input_dim=1)

#gpr.kernel.variance
#gpr.kernel.lengthscale
#gpr.noise
mean, cov=gpr.forward(Xnew, full_cov=False, noiseless=True)
#gpr = gp.models.GPRegression(X, y, kernel)
#GPregression(x,df_.iloc[2,:])
Xnew = torch.tensor([[2., 3, 1]])
gpr.forward(x_new)
p_value=scipy.stats.norm.cdf(300,loc=mean,scale=cov)

f_loc, f_cov = gpr(Xnew, full_cov=True)