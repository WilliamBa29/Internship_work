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
def rearrange(nopaststeps, data):
    """[Function which rearranges the dataframe, 'data', so that for each row, there is, in addition to the column corresponding to the current spot price, 
        a number of columns displaying the spot prices of the last 'nopaststeps', settlement intervals]

   Args:
       data (dataframe): A dataframe consisting of the spot prices for each settlement interval over a number of days and years.
       nopaststeps (int): An integer no bigger than the length of 'data', which indicates how far back one looks.
       In other words, 'nopaststeps' indicates the number of columns to be included in the dataframe to be produced by this function where
       each column displays the past spot price of some  settlement interval, earlier to the interval corresponding to the row in df_/current spot price]  
   Returns:
       [type]: [returns a dataframe, where each row details not only the spotprice mentioned in that row with the same row index in 'data' but
       also includes 'nopastteps' columns which display the 'spotprices' of the last 'nopasteps' settlement intervals].
   """
    x = np.empty((len(data) - 10, 10), dtype=object)
    for i in range(nopaststeps, len(data)):
        for j in range(0, nopaststeps):
            x[i - 10, j] = data.iloc[i - j, 2]
    return x
# DIR_BASE = 'D:/documents'
# FILE_HISTORY = os.path.join(DIR_BASE, "CapPrice.csv")
# df_=pd.read_csv(FILE_HISTORY)
# df_=df_.iloc[range(0,15000),:]
# rearrange(10,df_)
