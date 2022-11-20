import os
import sys
sys.path.insert(0,'D:\Documents\OneDrive\Documents\Data Science\MaxKelsen\csenergy-internship')
from src.helpers.slicingdata import formatting
from src.helpers.slicingdata import week
from src.helpers.slicingdata1 import rearrange
from src.helpers.frequency import relfreq
import pandas as pd
import torch
import numpy as np
from src.modelling.Gaussian_Process2 import splitdata2,GPenergy2
from src.modelling.basic_sparse_GP1 import inducing_point_gamma
from src.modelling.basic_sparse_GP1 import Gaussiansparse, Gaussiansparse1

from src.modelling.testing import testing
from src.helpers.splitdata3 import splitdata3


import numpy as np
import pyro
import pyro.contrib.gp as gp
import math
DIR_BASE='D:/Documents/OneDrive/Documents/Data Science/MaxKelsen/csenergy-internship/data/raw'
FILE_HISTORY1=os.path.join(DIR_BASE,"CapPrice.csv")
df_1=pd.read_csv(FILE_HISTORY1)
df_1=df_1.iloc[range(0,10000),:]
#m=df_.iloc[:,0].str[3:6]
#x=np.empty([len(df_),5],dtype=object)
#x[:,0]=df_.iloc[:,1]
#x[m=='01',1]=1
#x[m=='02',1]=2
#x[m=='03',1]=3
#x[m=='04',1]=4
#x[m=='05',1]=5
#x[m=='06',1]=6
#x[m=='07',1]=7
#x[m=='08',1]=8
#x[m=='09',1]=9
#x[m=='10',1]=10
#x[m=='11',1]=11
#x[m=='12',1]=12
#df_1=df_1.iloc[range(0,50000),:]
x1,x2,Y1,Y2,t1,t2,scaler1,scaler=splitdata3(df_1,'min',1)

rae=testing(x1,x2,Y1,Y2,48,df_1,scaler1,scaler)