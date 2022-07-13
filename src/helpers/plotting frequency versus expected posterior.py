from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import LogFormatter
z=[y[i] for i in range(0,len(x)) if x[i]!=0]
print(z)
fig = plt.figure()
ax=plt.subplot(111)
#fig,ax=plt.subplots()
ax.loglog(x,y,'.')
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