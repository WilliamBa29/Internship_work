#%%
import os
import sys
sys.path.insert(0,'D:\Documents\OneDrive\Documents\Data Science\MaxKelsen\csenergy-internship')
from src.helpers.slicingdata import formatting
from src.helpers.slicingdata import week
from src.helpers.slicingdata1 import rearrange
from src.helpers.frequency import relfreq
import pandas as pd
#%%

#%%

#%%
import numpy as np
DIR_BASE = 'D:/documents'
FILE_HISTORY = os.path.join(DIR_BASE, "CapPrice.csv")
df_=pd.read_csv(FILE_HISTORY)

s=np.empty(len(df_),dtype=object)
s[df_.iloc[:,3]>0]=1
s[df_.iloc[:,3]==0]=0
#df_=df_.iloc[range(0,15000),:]
#m=df_.iloc[:,0].str[3:6]
#arr = np.empty(len(df_), dtype=object)
#arr[m=='Jan']=1
#arr[m=='Feb']=2
#arr[m=='Mar']=3
#arr[m=='Apr']=4
#arr[m=='May']=5
#arr[m=='Jun']=6
#arr[m=='Jul']=7
#arr[m=='Aug']=8
#arr[m=='Sep']=9
#arr[m=='Oct']=10
#arr[m=='Nov']=11
#arr[m=='Dec']=12
#x=week(1,df_,arr)
#data2,E,s=formatting(df_,'Jan',2010)
#2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021
#,rearrange(10,df_)
rel_freqs_train,freqs_train=relfreq(df_,'Non Workdays',(2010,2011,2012,2013,2014,2015,2016),
            ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
rel_freqs_test,freqs_test=relfreq(df_,'Non Workdays',(2017, 2018, 2019, 2020),
            ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
#x,x1=relfreq(df_,'Non Workdays',(2010),('May','Jun','Jul','Aug','Sep','Oct'))
#print(E==0)

#%% md


#%%
#print((freqs_test))
#%%

#%%

from src.modelling.simple_dirichlet_model_adrian_rajkamal import posterior_mean
means=posterior_mean(freqs_train)

#%%
print(rel_freqs_test)
print(means)
#%%
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import LogFormatter
z=[y[i] for i in range(0,len(x)) if x[i]!=0]
print(z)
fig = plt.figure()
ax=plt.subplot(111)
#fig,ax=plt.subplots()
ax.loglog(rel_freqs_test,means, '.')
formatter = LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.4))
ax.xaxis.set_minor_formatter(formatter)
ax.yaxis.set_minor_formatter(formatter)
#FrmatStrFormatter("%.2f")
fig.suptitle('Exceedance Event Relative Frequencies For Each Interval Plotted Against the Exceedance Event Posterior Probability Expectation')
plt.xlabel('Exceedence Event Relative Frequencies')
plt.ylabel('Exceedance Event Posterior Probability Expectation')
#plt.tick_params(axis='y', which='minor')
#plt.tick_params(axis='x', which='minor')
#subsx=[0.5, 1.0, 1.5]
#ax.set_xscale('log')
#ax.set_yscale('log')
#ax.set_xticks([10^(-2)])
#ax.set_yticks([10^(-2)])


#%%
x=np.array([3,4,6])
x[[0,1,2]]=[2,1,2]
print(x)