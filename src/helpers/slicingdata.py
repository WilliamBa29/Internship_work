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
# df_=df_.iloc[range(0,100),:]
def dat(df_, Year1, E, j, c, s, work, t):
    """
   Helper function for formatting. This function produces one row of the workdays divided dataframe that is returned by the 'formatting' function.
   
   Args:
    df_(dataframe): [The original data set which contains the spot prices for each half hour interval]
    Year1 (array): [An array whose elements are the unique months and years mentioned in df_]
    E (array): [A vector of length equal to df_ which for each row/elements indicates whether
    the date corresponding to that row in df_ is a weekday or a weekend.
    A weekday is indicated by the value 1 and weekend day is indicated by the value 0]
    j (int): [An index which indicates the year corresponding to the date that the current row of the dataframe divided into workdays and weekend days relates to.
    j is not itself, a 'year', the 'year' of concern is the jth element of the 'Year1' array that is an input to this function].
    c (int): [An index which indicates the month corresponding to the data that the current row of the dataframe divided into workdays and weekend days relates to.
    c is not itselfr, a 'month', the 'month' of concern is the cth element of the 'Year1' array that is an input to the function].
    s (array): [An array of length equal to the length of df_ which for each row of df_ 
    has value 1 if that row of df_ corresponds to an exceedance event and 0 otherwise].
    work(string): [A string which is either 'Workdays' or 'Non Workdays'].
    t(int):       [An integer between 1 and 48 which indicates which settlement interval corresponds to the row to be returned by the function, 'dat']
   """

    return np.array([Year1[j], Year1[c], work, t, sum(s[(
                (df_.iloc[:, 0].str[7:9] == Year1[j][2:4]) & (
                    df_.iloc[:, 0].str[3:6] == Year1[c]) & (E == 1) & (
                            df_.iloc[:, 1] == t)).values])])


def week(rownumber: int, df_, arr):
    """[Function which determines whether a given day corresponds to a week day or a weekend day]

   Args:
       rownumber (int): [index which indicates the row number of the dataset, df_ which the date corresponds to]
       df_ ([dataframe]): [The dataset, whose 0th column includes a date]
       arr ([array]): [A vector with as many rows as df_ where the value of the ith row is number (between 1 and 12) 
       corresponding to the month included in the date on that row within the dataset df_. 
       The values corresponding to each month describes the months from January to December in increasing order (ie January is assigned 1 and December is assigned 12)]

   Returns:
       [type]: [returns a 1 if the date corresponding to the row (corresponding to the row number of df_) is a weekday and is a 0 if it is a weekend]
   """
    data = df_

    x = np.empty(len(data), dtype=object)
    # m=data.iloc[:,0].str[3:6]
    # arr[m=='Jan']=1
    # arr[m=='Feb']=2
    # arr[m=='Mar']=3
    # arr[m=='Apr']=4
    # arr[m=='May']=5
    # arr[m=='June']=6
    # arr[m=='July']=7
    # arr[m=='Aug']=8
    # arr[m=='Sep']=9
    # arr[m=='Oct']=10
    # arr[m=='Nov']=11
    # arr[m=='Dec']=12
    # datet=np.empty(len(data),dtype=object)
    # x = np.ctypeslib.as_array(shared_array)
    i = rownumber
    # for i in range(0,len(data)):

    # pass
    if data.iloc[i, 0][0] == 0:
        datet = datetime.date(int(f'20{data.iloc[i, 0][7:9]}'), arr[i], int(data.iloc[i, 0][1:2]))
    else:
        datet = datetime.date(int(f'20{data.iloc[i, 0][7:9]}'), arr[i], int(data.iloc[i, 0][0:2]))

    if (datet.weekday() == 5 or datet.weekday() == 6):
        x[i] = 0
    else:
        x[i] = 1
    return x[i]


def formatting(df_, month, year):
    """[Function which produces a new dataframe, data2, corresponding to a month and year of choice, which divides
 each interval into workdays and non-workdays]
 month asks for the first three letters of the desired month as a string (with a capital first letter), 
 or a tuple whose (multiple)entries are the first three letters of months each with a capital first letter, 
 written as strings

 Args:
     df_(dataframe): [The original dataset which indicates the spot price of electricity for a given date for each half hour interval]
     
     month (str/tuple): [The first three letters, if str, (where the first letter is capital) of the month
     for which a division into workday/non-workday data is desired
     or a tuple of string entries where each entry is the first three letters (first letter is capital) of the months for which division into workday and non-workday is desired].
     year(int/tuple): [The year or a tuple of years for which one wants the  corresponding data to be split into workday and non-workday]
 
 """

    s = np.empty(len(df_), dtype=object)
    s[df_.iloc[:, 3] > 0] = 1
    s[df_.iloc[:, 3] == 0] = 0
    m = df_.iloc[:, 0].str[3:6]
    arr = np.empty(len(df_), dtype=object)
    arr[m == 'Jan'] = 1
    arr[m == 'Feb'] = 2
    arr[m == 'Mar'] = 3
    arr[m == 'Apr'] = 4
    arr[m == 'May'] = 5
    arr[m == 'Jun'] = 6
    arr[m == 'Jul'] = 7
    arr[m == 'Aug'] = 8
    arr[m == 'Sep'] = 9
    arr[m == 'Oct'] = 10
    arr[m == 'Nov'] = 11
    arr[m == 'Dec'] = 12
    data1 = pd.DataFrame(columns=['Year', 'Month', 'Day_type', 'Interval', 'Exceedance_Frequency'])
    # data1=np.empty(98, dtype=object)
    Year = pd.DataFrame(columns=['Year', 'Month'])
    for i in range(0, len(df_)):
        Year = Year.append(pd.DataFrame([[f'20{df_.iloc[i, 0][7:9]}', df_.iloc[i, 0][3:6]]],
                                        columns=['Year', 'Month']))

    Year1 = np.unique(Year)
    f1 = np.zeros(len(Year1))
    if type(month) == str:

        f = (month == Year1)
        f = np.array(f)
        f = f.astype(int)
        f1 = f1 + f
    else:
        for g in range(0, len(month)):
            f = (month[g] == Year1)

            f = np.array(f)
            f = f.astype(int)
            f1 = f1 + f
    f1 = (f1 == 1)
    # t=(month==Year1)
    t = f1

    t1 = np.zeros(len(Year1))
    if type(year) == int:

        k = (f'{year}' == Year1)
        k = np.array(k)
        k.astype(int)
        t1 = t1 + k
    else:
        for g1 in range(0, len(year)):
            k = (f'{year[g1]}' == Year1)
            k = np.array(k)
            k.astype(int)
            t1 = t1 + k
    t1 = (t1 == 1)
    M = [i for i, x in enumerate(t) if x]
    M1 = [i for i, x in enumerate(t1) if x]
    # x = np.ctypeslib.as_ctypes(np.zeros((len(df_))))
    # shared_array = sharedctypes.RawArray(x._type_, x)

    P = Pool(os.cpu_count() - 1)
    a_arg = range(0, len(df_))
    thirdarg = arr
    second_arg = df_
    # c=list(zip(a_arg, repeat(second_arg),repeat(thirdarg)))
    b, c = df_, arr
    c1 = np.zeros(len(df_))
    if type(year) == int:

        k = ((str(year)[2:4]) == df_.iloc[:, 0].str[7:9])
        k = np.array(k)
        k.astype(int)
        c1 = c1 + k
    else:
        for r in range(0, len(year)):
            f = (str(year[r])[2:4] == df_.iloc[:, 0].str[7:9])

            f = np.array(f)
            f = f.astype(int)
            c1 = c1 + f
    # c1=(c1==1)

    c2 = np.zeros(len(df_))
    if type(month) == str:

        f = (month == df_.iloc[:, 0].str[3:6])
        f = np.array(f)
        f = f.astype(int)
        c2 = c2 + f
    else:
        for r in range(0, len(month)):
            f = (month[r] == df_.iloc[:, 0].str[3:6])

            f = np.array(f)
            f = f.astype(int)
            c2 = c2 + f
    c3 = c2 * c1
    c3 = (c3 == 1)
    M2 = [i for i, x in enumerate(c3) if x]
    # t=(month==Year1)

    # df_.iloc[df.iloc[:,0].str[7:9]]
    arguments = [(a, b, c) for a in M2]
    E = np.zeros(len(df_))
    #  print(arguments[-1])
    # week(1,df_,arr)
    # print(c[3][0])
    E1 = np.array(P.starmap(week, arguments))
    E[M2] = E1
    # E=np.ctypeslib.as_array(shared_array)

    l = 0
    # print(M1)
    # print(M)
    for j in M1:

        for c in M:
            arg1, arg2, arg3, arg4, arg5, arg6, arg7 = df_, Year1, E, j, c, s, 'Workdays'
            arguments1 = [(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8) for arg8 in range(1, 49)]

            data1 = np.array(P.starmap(dat, arguments1))
            arg1, arg2, arg3, arg4, arg5, arg6, arg7 = df_, Year1, E + 1, j, c, s, 'Non Workdays'
            arguments1 = [(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8) for arg8 in range(1, 49)]
            if l == 0:

                data2 = data1
            else:
                data2 = np.concatenate((data2, data1))

            data1 = np.array(P.starmap(dat, arguments1))
            data2 = np.concatenate((data2, data1))
            l = l + 1  # pd.DataFrame([[Year1[j],Year1[c],'Work days',t,sum(df_.iloc[((df_.iloc[:,0].str[7:9]==Year1[j][2:4]) & (df_.iloc[:,0].str[3:6]==Year1[c]) & (E==1)).values,3])]],columns=['Year','Month', 'Day_type', 'Interval','Exceedance_Frequency']))
            # data1=data1.append(pd.DataFrame([[Year1[j],Year1[c],'Work days',t,sum(df_.iloc[((df_.iloc[:,0].str[7:9]==Year1[j][2:4]) & (df_.iloc[:,0].str[3:6]==Year1[c]) & (E==1)).values,3])]],columns=['Year','Month', 'Day_type', 'Interval','Exceedance_Frequency']))
            # print(((df_.iloc[:,0].str[7:9]==Year1[j][2:4]) & (df_.iloc[:,0].str[3:6]==Year1[c]) & (E==1)).values)
            # print(len(pd.DataFrame([[Year1[j],Year1[c],'Work days',t,sum(df_.iloc[((df_.iloc[:,0].str[7:9]==Year1[j][2:4]) & (df_.iloc[:,0].str[3:6]==Year1[c]) & (E==1)).values,3])]],columns=['Year','Month', 'Day_type', 'Interval','Exceedance_Frequency'])))
            # for t in range(1,49):

            # data1=data1.append(pd.DataFrame([[Year1[j],Year1[c],'Non Work days',t,sum(df_.iloc[((df_.iloc[:,0].str[7:9]==Year1[j][2:4]) & (df_.iloc[:,0].str[3:6]==Year1[c]) & (E==0)).values,3])]],columns=['Year','Month', 'Day_type', 'Interval','Exceedance_Frequency']))

    data2 = pd.DataFrame(data2,
                         columns=['Year', 'Month', 'Day_type', 'Interval', 'Exceedance_Frequency'])
    np.savetxt('D:/documents/data2.csv', data2, delimiter=',', fmt='%s',
               header="Year, Month, Day_type, Interval, Exceedance_Frequency")  # saves dataframe to excel file
    return data2, E, s


# with open('data1.csv', mode='w') as data1:
#   employee_writer = csv.writer(data1, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
if __name__ == "__main__":
    DIR_BASE = 'D:/documents'
    FILE_HISTORY = os.path.join(DIR_BASE, "CapPrice.csv")
    df_ = pd.read_csv(FILE_HISTORY)
    df_ = df_.iloc[range(0, 8000), :]
    formatting(df_, ('Jan', 'Jul'), 2010)
    # formatting(df_,'Jan',2010)
