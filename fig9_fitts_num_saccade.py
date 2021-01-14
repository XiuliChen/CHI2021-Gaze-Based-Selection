from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# the number of saccade in D=5 and D=10 conditions
num_saccade_D5=np.array([1.6325, 1.31,   1.1275, 1.02,   1.,     1.    ])
num_saccade_D10=np.array([2.0425, 1.825,  1.5625, 1.2275, 1.0825, 1.0275])

plt.figure(figsize=(8,4.5))
plt.subplot(1,2,1)


def ID(W,D):
	return np.log2(2*D/W)

print(ID(15,255))
print(ID(45,64))



W= np.array([1,1.5,2,3,4,5])
print(ID(W,5))
print(ID(W,10))
xxx

ID1=ID(W,5)
ID2=ID(W,10)

plt.plot(ID1,num_saccade_D5,'ro-',label='Distance=5')
plt.plot(ID2,num_saccade_D10,'bs--',label='Distance=10')

plt.xlabel('Index of difficulty (bits) ',fontsize=14)
plt.ylabel('Saccades per trial',fontsize=14)
plt.ylim([0.9,2.4])
plt.title('Model',fontsize=14)
plt.legend(fontsize=12)

# plot the data

plt.subplot(1,2,2)
ids=[]
for dd in np.array([5,10]):
	for tmp in W:
		ids.append(ID(tmp,dd))

ids=np.array(ids)
ids=np.unique(ids)
print(ids)

unit1=330
up=np.array([32,36,40,36,88,135,177,216,264])
mid=np.array([18,27,26,24,68,101,140,183,218])

mid1=mid/unit1+1
up1=(up-mid)/unit1

plt.errorbar(ids,mid1,yerr=up1,color='k',fmt='s-',label='Averaged (D5 and D10)')

#plt.xticks(np.round(ids,2))
plt.xlabel('Index of difficulty (bits) ',fontsize=14)
plt.ylabel('Saccades per trial',fontsize=14)
plt.ylim([0.9,2.4])
plt.title('Data',fontsize=14)

plt.legend(fontsize=12)


plt.savefig('figures/fitss_saccade_num.pdf')




	
	
		



