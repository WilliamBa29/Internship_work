import scipy
import csv
import numpy as np
import pandas as pd
import datetime
from multiprocessing import Pool
from multiprocessing import sharedctypes
from itertools import repeat
import os
import itertools
# DIR_BASE = 'D:/documents'
# FILE_HISTORY = os.path.join(DIR_BASE, "CapPrice.csv")
# df_=pd.read_csv(FILE_HISTORY)
import os
import sys

sys.path.insert(0, 'D:\Documents\OneDrive\Documents\Data Science\MaxKelsen\csenergy-internship')
from src.helpers.slicingdata import formatting


# DIR_BASE = 'D:/documents'
# FILE_HISTORY = os.path.join(DIR_BASE, "CapPrice.csv")
# df_=pd.read_csv(FILE_HISTORY)
def relfreq(df_, work, year, month):
    """[Function which returns the relative frequencies, and frequencies of exceedence events
      in each interval for the set of either 'workdays' or 'non-workdays' for a specific month and year]

   Args:
       df_ (dataframe): [A dataframe consisting of the spot prices for each settlement interval over a number of days and years].
       work (string): [A string which is either 'Workdays', or 'Non Workdays]
       
       year (int/tuple): [The year for which one wants to calculate these relative frequencies.
       Can be an individual year, or a tuple of years].
       month(string/tuple): [The month for which one wants to calculate these relative frequencies. 
       Can be an individual month, or a tuple of months. 
       Each month is represented by a string]. 
   Returns:
       [type]: [returns a 48 element array, where the ith element details the relative frequency of the ith settlement interval].
   """
    data2, E, s = formatting(df_, month, year)
    G = np.zeros(len(df_))
    # print(len(df_.iloc[((month==df_.iloc[:,0].str[3:6])&((str(year))[2:4]==df_.iloc[:,0].str[7:9])).values])/48)
    f1 = np.zeros(len(df_))
    if type(month) == str:

        f = (month == df_.iloc[:, 0].str[3:6])
        f = np.array(f)
        f = f.astype(int)
        f1 = f1 + f
    else:
        for g in range(0, len(month)):
            f = (month[g] == df_.iloc[:, 0].str[3:6])

            f = np.array(f)
            f = f.astype(int)
            f1 = f1 + f
    f1 = (f1 == 1)
    # t=(month==Year1)
    g1 = np.zeros(len(df_))
    if type(year) == int:

        k = ((str(year))[2:4] == df_.iloc[:, 0].str[7:9])
        k = np.array(k)
        k.astype(int)
        g1 = g1 + k
    else:
        for c in range(0, len(year)):
            k = ((str(year[c]))[2:4] == df_.iloc[:, 0].str[7:9])
            k = np.array(k)
            k.astype(int)
            g1 = g1 + k
    g1 = (g1 == 1)
    # print(sum(g1.astype(int)))

    # for i in range(1,int(len(df_.iloc[((f1)&(g1))])/48)):

    # if i<=9:
    # G[((f1)&(g1)&(f'0{i}'==df_.iloc[:,0].str[0:2])).values]=sum(s[((f1)&(g1)&(f'0{i}'==df_.iloc[:,0].str[1])).values])
    # print(sum(s[((month==df_.iloc[:,0].str[3:6])&(g1)&(f'0{i}'==df_.iloc[:,0].str[1])).values]))
    # else:
    # print(sum(s[((month==df_.iloc[:,0].str[3:6])&((str(year))[2:4]==df_.iloc[:,0].str[7:9])&(str(i)==df_.iloc[:,0].str[0:2])).values]))
    # G[((f1)&(g1)&(str(i)==df_.iloc[:,0].str[0:2]))]=sum(s[((f1)&(g1)&(str(i)==df_.iloc[:,0].str[0:2])).values])
    # print(data2)
    # arr = np.empty(len(df_), dtype=object)
    # arr[m=='Jan']=1
    # arr[m=='Feb']=2
    # arr[m=='Mar']=3
    # arr[m=='Apr']=4
    # arr[m=='May']=5
    # arr[m=='Jun']=6
    # arr[m=='Jul']=7
    # arr[m=='Aug']=8
    # arr[m=='Sep']=9
    # arr[m=='Oct']=10
    # arr[m=='Nov']=11
    # arr[m=='Dec']=12
    # a_arg=range(0,len(df_))
    # thirdarg=arr
    # second_arg=df_
    # c=list(zip(a_arg, repeat(second_arg),repeat(thirdarg)))
    # b,c=df_,arr
    # arguments=[(a,b,c) for a in range(0,len(df_))]
    #  print(arguments[-1])
    # week(1,df_,arr)
    # print(c[3][0])
    # E=np.array(P.starmap(
    # week,
    # arguments))

    freq = np.empty(48, dtype=object)
    freq1 = np.empty(48, dtype=object)
    f2 = np.zeros(len(data2))
    if type(month) == str:

        f = (month == data2.iloc[:, 1])
        f = np.array(f)
        f = f.astype(int)
        f2 = f2 + f
    else:
        for g in range(0, len(month)):
            f = (month[g] == data2.iloc[:, 1])

            f = np.array(f)
            f = f.astype(int)
            f2 = f2 + f
    f2 = (f2 == 1)
    # t=(month==Year1)

    # month==data2.iloc[:,1]
    g2 = np.zeros(len(data2))
    if type(year) == int:

        k = (str(year) == data2.iloc[:, 0])
        k = np.array(k)
        k.astype(int)
        g2 = g2 + k
    else:
        for c in range(0, len(year)):
            k = (str(year[c]) == data2.iloc[:, 0])
            k = np.array(k)
            k.astype(int)
            g2 = g2 + k
    g2 = (g2 == 1)

    # print((month==df_.iloc[:,0].str[3:6]))
    # print(df_.iloc[:,0].str[7:9].astype(int))
    # print(month==data2.iloc[:,1])
    # print(str(year)==data2.iloc[:,0])
    # print(work==data2.iloc[:,2])
    # print(type(data2.iloc[0,3]))
    # i=1
    # print(((month==data2.iloc[:,1])&(str(year)==data2.iloc[:,0])&(work==data2.iloc[:,2])&(str(i)==data2.iloc[:,3])).values)
    # print((data2.iloc[((month==data2.iloc[:,1])&(str(year)==data2.iloc[:,0])&(work==data2.iloc[:,2])&(str(i)==data2.iloc[:,3])).values,4]))
    # print((data2.iloc[0,4]))
    # print(type(data2.iloc[0,0]))
    # print(type(year))
    # (work==data2.iloc[:,02])&(1==data2.iloc[:,3])))
    # print((str(year))[2:4]==df_.iloc[:,0].str[7:9])
    # print(sum((G>=1).astype(int)))
    # print(sum((s==1).astype(int)))
    print(sum(f2.astype(int)))
    print(sum(g2.astype(int)))
    print(sum((work == data2.iloc[:, 2]).astype(int)))
    for i in range(1, 49):

        if work == 'Workdays':
            # print(((data2.iloc[((month==data2.iloc[:,1])&(str(year)==data2.iloc[:,0])&(work==data2.iloc[:,2])&(str(i)==data2.iloc[:,3])).values,4]).astype(int))/len(df_.iloc[((month==df_.iloc[:,0].str[3:6])& ((str(year))[2:4]==df_.iloc[:,0].str[7:9])&(i==df_.iloc[:,1])&(E==0)).values]))
            # print(len(df_.iloc[((month==df_.iloc[:,0].str[3:6])& ((str(year))[2:4]==df_.iloc[:,0].str[7:9])&(i==df_.iloc[:,1])&(E==0)).values]))
            # print(i==df_.iloc[:,1])
            # print((month==df_.iloc[:,0].str[3:6]))
            # print(((str(year))[2:4]==df_.iloc[:,0].str[7:9]))
            # print(sum(E==0))
            # i==df_.iloc[:,1]
            # print(len(df_.iloc[((month==df_.iloc[:,0].str[3:6])& ((str(year))[2:4]==df_.iloc[:,0].str[7:9])&(i==df_.iloc[:,1])&(E==0)).values]))
            # print(len(df_.iloc[((f1)& (g1)&(i==df_.iloc[:,1])&(E==0)&(G>=1)).values]))

            freq[i - 1] = (sum((data2.iloc[((f2) & (g2) & (work == data2.iloc[:, 2]) & (
                        str(i) == data2.iloc[:, 3])).values, 4]).astype(int))) / len(
                df_.iloc[((f1) & (g1) & (s == 1) & (E == 1))])

            freq1[i - 1] = (sum((data2.iloc[((f2) & (g2) & (work == data2.iloc[:, 2]) & (
                        str(i) == data2.iloc[:, 3])).values, 4]).astype(int)))
            # print(len(df_.iloc[((month==df_.iloc[:,0].str[3:6])& (year==df_.iloc[:,0].str[7:9].astype(int))&(i==df_.iloc[:,1])&(E==0)).values]))
        if work == 'Non Workdays':
            # print(len(df_.iloc[((f1)& (g1)&(i==df_.iloc[:,1])&(E==1)&(G>=1)).values]))
            freq1[i - 1] = (sum((data2.iloc[((f2) & (g2) & (work == data2.iloc[:, 2]) & (
                        str(i) == data2.iloc[:, 3])).values, 4]).astype(int)))
            freq[i - 1] = (sum((data2.iloc[((f2) & (g2) & (work == data2.iloc[:, 2]) & (
                        str(i) == data2.iloc[:, 3])).values, 4]).astype(int))) / len(
                df_.iloc[((f1) & (g1) & (s == 1) & (E == 0))])

    return freq, freq1
