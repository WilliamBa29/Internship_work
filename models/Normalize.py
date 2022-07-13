def prob(p_value):
 
 D=np.floor(len(p_value)/48)
 D1 = np.empty(D, dtype=object)#plt.plot(model=gpr, plot_observed_data=True, plot_predictions=True)
 R=len(p_value)%48
 for i in range(0,D):
     C=np.empty(48,dtype=object)
     sum=0
     for j in range(0,48):
        C[j]=p_value[j]
        sum=sum+p_value[j]
     C=C/sum
     D1[R+i*D]=C   