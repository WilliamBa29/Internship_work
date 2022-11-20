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
#DIR_BASE = 'D:/documents'
#FILE_HISTORY = os.path.join(DIR_BASE, "CapPrice.csv")
#df_=pd.read_csv(FILE_HISTORY)
def relfreq(work, year, month):
    