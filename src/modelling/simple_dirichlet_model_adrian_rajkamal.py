import torch
import pandas as pd

"""
    Simple Dirichlet Model using uninformative Dir(1, 1, ..., 1) prior with Multinomial 
    likelihood. Suppose an exceedance event occurred. This distribution models the probability 
    that the exceedance event that had occurred, occurred in settlement interval i, for all i in 
    {1, ..., 48}.
    
    Multinomial Likelihood considers number of exceedance events in interval i, within some time 
    period.
    
    Replace 'jan_exceedances' with the desired time period to be investigated.

    Author: Adrian Rajkamal
    Date: 6/12/2021
"""


def posterior_mean(data):
    """
        Computes Dirichlet posterior mean, assuming an uninformed (improper) prior, with Multinomial
        data.

        :param data: exceedance event counts for each settlement interval in some given time
        period (e.g. January 2021)

        :return: Joint posterior mean vector
    """
    posterior = data + 1  # (p | x) ~ Dir(x_1 + 1, ..., x_48 + 1)
    total = float(sum(posterior))
    length = int(len(posterior))
    return [float(posterior[i] / total) for i in range(length)]


""""""""""""""" EXAMPLE USAGE """""""""""""""
# num_intervals = 48
# df = pd.read_csv("../../data/processed/historical_spot_price.csv")
#
# # Consider exceedance events across January (for all years 2010 - 2021)
# jan_exceedances = df.loc[df.Date.str.contains("Dec") & df.Exceedance > 0., 'Settlement_Interval']
#
# # Get the total number of exceedance events for each interval
# counts = torch.zeros(num_intervals)
# for interval in jan_exceedances:
#    counts[interval] += 1
#
# Can compare below output with relative frequencies. Below output converges to the MLE,
# i.e. the relative frequencies, so compare for large datasets (e.g. like jan_exceedances)
#print(posterior_mean(counts))
