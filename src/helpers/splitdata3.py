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
def splitdata3(df_,t):

  """
  Helper function for the Gaussian Process model. This function creates both the training and validation dataset array of predictor variables (six of them) to be used in the model as X.
  This function also produces the training dataset vector and the validation dataset vector consisting of the values of the response variable (spot price)

  Args:

  df_(dataframe): The original dataframe that contains the spot prices and settlement interval numbers of each half hour interval.

  t(string): This is a variable that either takes the value 'min' in which cases the X data set undergoes max-min normalization
  or 'stand' in which case the X dataset produced undergoes standardization.

  Returns
   x1(Array): The training dataset array comprised of the six predictor variables.
   x2(Array): The validation dataset array comprised of the six predictor variables.
   Y1(Array): A vector consisting of the training dataset spot prices to be used as the training response variable.
   Y2(Array): A vector consisting of the validation dataset spot prices to be used as the validation response variable.
   """
  
  m=df_.iloc[:,0].str[3:6]
  x=np.empty([len(df_),6],dtype=object)
  Y=np.empty([len(df_),1],dtype=object)
  Y=df_.iloc[:,2]
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
     x[i,3]=int(f'20{df_.iloc[i,0][7:9]}')
     if df_.iloc[i,0][0]==0:
      print(int(f'20{df_.iloc[i,0][7:9]}'))
      print(x[i,1])
      print(int(df_.iloc[i,0][0:2]))
      x[i,2]=datetime.date(int(f'20{df_.iloc[i,0][7:9]}')    ,x[i,1], int(df_.iloc[i,0][1:2])).weekday()
     else:
      #print(int(f'20{df_.iloc[i,0][7:9]}'))
      #print(x[i,1])
      #print(int(df_.iloc[i,0][0:2]))

      x[i,2]=datetime.date(int(f'20{df_.iloc[i,0][7:9]}')   ,x[i,1], int(df_.iloc[i,0][0:2])).weekday() 
      x[i,4]=np.mean(df_.iloc[range(i-30,i),2]) 
      x[i,5]=np.mean(df_.iloc[range(i-30,i),3])
       
  x1=x[range(0,math.floor(0.8*len(x))),:]
  x2=x[range(math.floor(0.8*len(x)),len(x)),:]  
  Y1=Y[range(0,math.floor(0.8*len(x)))]
  Y2= Y[range(math.floor(0.8*len(x)),len(x))]
  Y1=Y1.values.reshape(1,-1)
  Y2=Y2.values.reshape(1,-1)
  Y1=np.transpose(Y1)
  Y2=np.transpose(Y2)
  if t=='min':
     scaler = MinMaxScaler()
     scaler.fit(x1) 
     x1=scaler.transform(x1) 
     x2=scaler.transform(x2)
     scaler=MinMaxScaler()
     scaler.fit(Y1)
     Y1=scaler.transform(Y1)
     Y2=scaler.transform(Y2)
  if t=='stand':
     scaler = StandardScaler()
     scaler = scaler.fit(x1)
     x1=scaler.transform(x1)
     x2=scaler.transform(x2)
     scaler = StandardScaler()
     scaler.fit(Y1)
     Y1=scaler.transform(Y1)
     Y2=scaler.transform(Y2)
  return(x1,x2,Y1, Y2)


