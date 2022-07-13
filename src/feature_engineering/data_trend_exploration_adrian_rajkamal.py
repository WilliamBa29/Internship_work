import pandas as pd

"""
    Some code to tabulate/list specific subsets of the data. Purely exists as a file to copy 
    pandas code from.
    
    Author: Adrian Rajkamal
    Date: 14/12/2021
"""

df = pd.read_csv("../../data/processed/historical_spot_price.csv")
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

exceedances = [[0 for i in range(12)] for j in range(len(months))]

# Get #exceedance events for each month, for each year
for i in range(len(months)):
    for j in range(12):
        month_year = months[j] + "-" + str(i + 10)
        exceedances[i][j] = df.loc[
            df.Date.str.contains(month_year) & df.Exceedance > 0., 'Settlement_Interval'].size

# January 2013 was interesting - probably temperature
for i in range(1, 32):
    print(i, df.loc[
        df.Date.str.match(f"{i}-Jan-11") & df.Exceedance > 0., 'Settlement_Interval'].size)

# May 2021 was particularly unusual across the board - solar rebate
for i in range(1, 32):
    print(i, df.loc[
        df.Date.str.match(f"{i}-May-21") & df.Exceedance > 0., 'Settlement_Interval'].size)

print(i, df.loc[
        df.Date.str.contains(f"-20") & df.Exceedance > 0., 'Settlement_Interval'].size)
