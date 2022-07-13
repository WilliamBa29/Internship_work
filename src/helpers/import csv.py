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
from src.helpers.slicingdata2 import week
from src.helpers.slicingdata1 import rearrange
from src.helpers.frequency import relfreq
import matplotlib.pyplot as plt
import torch

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
def splitdata3(df_):
    x=np.empty([len(df_),5],dtype=object)
    x[:,0]=df_.iloc[:,1]
    x[m=='01',1]=1
    x[m=='02',1]=2
    x[m=='03',1]=3
    x[m=='04',1]=4
    x[m=='05',1]=5
    x[m=='06',1]=6
    x[m=='07',1]=7
    x[m=='08',1]=8
    x[m=='09',1]=9
    x[m=='10',1]=10
    x[m=='11',1]=11
    x[m=='12',1]=12
    for i in range(0,len(df_)):
     x[i,3]=int(f'20{df_.iloc[i,0][7:9]}')
     if df_.iloc[i,0][0]==0:
      x[i,2]=datetime.date(int(f'20{df_.iloc[i,0][7:9]}')    ,x[i,1], int(df_.iloc[i,0][1:2]))
     else:
      x[i,2]=datetime.date(int(f'20{data.iloc[i,0][7:9]}')   ,x[i,1], int(df_.iloc[i,0][0:2])) 
     x[i,4]=np.mean(df_.iloc[range(i-30,i),2]) 
     x[i,5]=np.mean(df_.iloc[range(i-30,i),3])  
       
