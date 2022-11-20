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
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from itertools import repeat

import itertools
sys.path.insert(0,'..\..')
#sys.path.insert(0,'D:\Documents\OneDrive\Documents\Data Science\MaxKelsen\csenergy-internship')
from src.helpers.slicingdata import formatting
from src.helpers.slicingdata2 import week
from src.helpers.slicingdata1 import rearrange
from src.helpers.frequency import relfreq
import matplotlib.pyplot as plt
import torch

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
from sklearn.preprocessing import OneHotEncoder
def splitdata3(df_,d,month, proptrain,propvalidation):

  """
  Helper function for the Gaussian Process model. This function creates both the training and validation dataset array of predictor variables (six of them) to be used in the model as X.
  This function also produces the training dataset vector and the validation dataset vector consisting of the values of the response variable (spot price)

  Args:

  df_(dataframe): The original dataframe that contains the spot prices and settlement interval numbers of each half hour interval.

  t(string): This is a variable that either takes the value 'min' in which cases the X data set undergoes max-min normalization
  or 'stand' in which case the X dataset produced undergoes standardization.
  month(int): A number from 1-12 which designates the month for which one wants data 
  (months are denoted such number in accordance with their temporal order. Ie, January is designated by 1 and December by 12)
  Returns
   x1(Array): The training dataset array comprised of the six predictor variables.
   x2(Array): The validation dataset array comprised of the six predictor variables.
   Y1(Array): A vector consisting of the training dataset spot prices to be used as the training response variable.
   Y2(Array): A vector consisting of the validation dataset spot prices to be used as the validation response variable.
   """
  
  m=df_.iloc[:,0].str[3:6]
  x=np.empty([len(df_),12],dtype=object)
  Y=np.empty([len(df_),1],dtype=object)
  Y=df_.iloc[:,2].to_numpy()
  x[:,0]=df_.iloc[:,1]
  x[m=='Jan',1]=1
  x[m=='Feb',1]=2
  x[m=='Mar',1]=3
  x[m=='Apr',1]=4
  x[m=='May',1]=5
  x[m=='Jun',1]=6
  x[m=='Jul',1]=7
  x[m=='Aug',1]=8
  x[m=='Sep',1]=9
  x[m=='Oct',1]=10
  x[m=='Nov',1]=11
  x[m=='Dec',1]=12
  for i in range(0,len(df_)):
     x[i,9]=int(f'20{df_.iloc[i,0][7:9]}')
     if df_.iloc[i,0][0]==0:
      #print(int(f'20{df_.iloc[i,0][7:9]}'))
      #print(x[i,1])
     # print(int(df_.iloc[i,0][0:2]))
      x[i,2]=datetime.date(int(f'20{df_.iloc[i,0][7:9]}')    ,x[i,1], int(df_.iloc[i,0][1:2])).weekday()
     else:
      #print(int(f'20{df_.iloc[i,0][7:9]}'))
      #print(x[i,1])
      #print(int(df_.iloc[i,0][0:2]))

      x[i,2]=datetime.date(int(f'20{df_.iloc[i,0][7:9]}')   ,x[i,1], int(df_.iloc[i,0][0:2])).weekday() 
     
     x[i,10]=np.mean(df_.iloc[range(i-30,i),2]) 
     x[i,11]=np.mean(df_.iloc[range(i-30,i),3])
  for i in range(0,len(df_)):
    if x[i,2]==0:
       x[i,2]==1
    else:
       x[i,2]==0
    if x[i,2]==1:
      x[i,3]==1
    else:
      x[i,3]=0
    if x[i,2]==2:
      x[i,4]==1
    else:
      x[i,4]=0
    if x[i,2]==3:
      x[i,5]==1
    else:
      x[i,5]=0
    if x[i,2]==4:
      x[i,6]==1
    else:
      x[i,6]==0
    if x[i,2]==5:
      x[i,7]==1
    else:
      x[i,7]==0
    if x[i,2]==6:
      x[i,8]==1
    else:
      x[i,8]=0
  #print(len(x))

  #enc = OneHotEncoder(handle_unknown='ignore',categories=[range(1,8)],sparse=False)
  #enc.fit(x[:,2].reshape(1,-1))
  #x[:,range(2,9)]=enc.transform(x[:,2].reshape(1,-1)) 
  #print( x[:,range(2,9)]) 
  Y=Y[x[:,1]==month]
  x=x[x[:,1]==month,:]     
  x1=x[range(0,math.floor(proptrain*len(x))),:]
  x2=x[range(math.floor(proptrain*len(x)),math.floor((proptrain+propvalidation)*len(x))),:]
  x3=x[range(math.floor((proptrain+propvalidation)*len(x)),len(x)),:]#print(len(x1)) 
  print('hello')
  print(len(x3))
  
  #t=[range(0,math.floor(0.8*len(x)))]
  #print(x2)
  t=np.empty(math.floor(proptrain*len(x)))
  for i in range(0, math.floor(proptrain*len(x))):
    t[i]=i+1
  t1=np.empty(math.floor((proptrain+propvalidation)*len(x))-math.floor(proptrain*len(x)))
  for j in range(math.floor(proptrain*len(x)),math.floor((proptrain+propvalidation)*len(x))):
    t1[j-math.floor(proptrain*len(x))] =j+1
  
  
  t2=np.empty(len(x)-math.floor((proptrain+propvalidation)*len(x)))  
  for z in range(math.floor((proptrain+propvalidation)*len(x)),len(x)):
    t2[z-math.floor((proptrain+propvalidation)*len(x))]=z+1
  #t=t.reshape(1,-1)
  #t1=t1.reshape(1,-1)
  t=torch.tensor(t).float()
  t1=torch.tensor(t1).float()
  t2=torch.tensor(t2).float()
  #print(len(Y))
  #print(math.floor(0.8*len(x)))
  Y1=Y[range(0,math.floor(proptrain*len(x)))]
  Y2=Y[range(math.floor(proptrain*len(x)),math.floor((proptrain+propvalidation)*len(x)))]
  Y3=Y[range(math.floor((proptrain+propvalidation)*len(x)),len(x))]#pr
  
 # Y1=Y[range(0,math.floor(0.8*len(x)))]

 # Y2= Y[range(math.floor(0.8*len(x)),len(x))]
  Y1=Y1.reshape(1,-1)
  Y2=Y2.reshape(1,-1)
  Y3=Y3.reshape(1,-1)
  Y1=np.transpose(Y1)
  Y2=np.transpose(Y2)
  Y3=np.transpose(Y3)
  if d=='min':
     scaler = MinMaxScaler()
     scaler.fit(x1[[0,1,9,10,11],:]) 
     x1[[0,1,9,10,11],:]=scaler.transform(x1[[0,1,9,10,11],:]) 
     x2[[0,1,9,10,11],:]=scaler.transform(x2[[0,1,9,10,11],:])
     x3[[0,1,9,10,11],:]=scaler.transform(x3[[0,1,9,10,11],:])
     scaler1=MinMaxScaler()
     scaler1.fit(Y1)
     Y1=scaler1.transform(Y1)
     Y2=scaler1.transform(Y2)
     Y3=scaler1.transform(Y3)
  if d=='stand':
     scaler = StandardScaler()
     scaler = scaler.fit(x1)
     x1[[0,1,9,10,11],:]=scaler.transform(x1[[0,1,9,10,11],:])
     x2[[0,1,9,10,11],:]=scaler.transform(x2[[0,1,9,10,11],:])
     x3[[0,1,9,10,11],:]=scaler.transform(x3[[0,1,9,10,11],:])
     scaler1 = StandardScaler()
     scaler1.fit(Y1)
     Y1=scaler1.transform(Y1)
     Y2=scaler1.transform(Y2)
     Y3=scaler1.transform(Y3)
  if d=='none':
    scaler=1
    scaler1=2

  x1=x1.astype(float)
  x2=x2.astype(float)
  #print(np.isnan(x1))
  return(x1,x2,x3,Y1,Y2,Y3,t,t1,t2,scaler1,scaler)


