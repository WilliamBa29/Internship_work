import os
import sys

sys.path.insert(0, 'D:\Documents\OneDrive\Documents\Data Science\MaxKelsen\csenergy-internship')
from src.helpers.slicingdata import formatting
from src.helpers.slicingdata import week
from src.helpers.slicingdata1 import rearrange
from src.helpers.frequency import relfreq
import pandas as pd
import torch
import numpy as np
from src.modelling.Gaussian_Process2 import splitdata2, GPenergy2
import numpy as np
import pyro
import pyro.contrib.gp as gp
import math

DIR_BASE = 'D:/Documents/OneDrive/Documents/Data Science/MaxKelsen/csenergy-internship/data/raw'
FILE_HISTORY1 = os.path.join(DIR_BASE, "Demand.csv")
df_1 = pd.read_csv(FILE_HISTORY1)
# df_1=df_1.iloc[range(0,50000),:]
x2, data1 = splitdata2(df_1, '01', 'Workdays')
# print(data1)
data2 = data1.iloc[range(0, math.floor(0.8 * len(data1))), 0]
x = x2[range(0, math.floor(0.8 * len(data1)))]
x1 = x2[range(math.floor(0.8 * len(data1)), len(data1))]
# x=np.concatenate(x,data1.iloc[range(0,math.floor(0.8*len(data1))),1])
# x1=np.concatenate(x1,data1.iloc[range(math.floor(0.8*len(data1)),len(data1)),1])
x = x.astype(float)
x1 = x1.astype(float)
data2 = data2.astype(float)
# validationdata=validationdata.astype(float)
print(type(x1))
validationdata = data1.iloc[range(math.floor(0.8 * len(data1)), len(data1)), 0]
validationdata = validationdata.astype(float)
# kernel=gp.kernels.RBF(input_dim=1,variance=torch.tensor(10000.),lengthscale=torch.tensor(100.))
kernel = gp.kernels.Periodic(input_dim=1, variance=torch.tensor(10000.))
# kernel1= gp.kernels.Sum(gp.kernels.Periodic(input_dim=1),gp.kernels.Periodic(input_dim=1))
# kernel=gp.kernels.Sum(kernel1,gp.kernels.Periodic(input_dim=1))
# print(x1)
# print(x1)
# print(data2)
# print(validationdata)
lower, upper, m1, m, cov, p, Rsq = GPenergy2(x, x1, data2, validationdata, kernel,
                                             meanf=lambda x: 7000)
x = np.transpose(np.vstack((x, m1.detach().numpy())))
x1 = np.transpose(np.vstack((x1, m.detach().numpy())))
data3 = data1.iloc[range(0, math.floor(0.8 * len(data1))), 1]
validationdata1 = data1.iloc[range(math.floor(0.8 * len(data1)), len(data1)), 1]
kernel = gp.kernels.Periodic(input_dim=1, variance=torch.tensor(20.), lengthscale=torch.tensor(1.))
lower1, upper1, m2, m3, cov1, p1, Rsq1 = GPenergy2(x, x1, data3, validationdata1, kernel,
                                                   meanf=lambda x: 0.2)
