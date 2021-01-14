
import numpy as np
import matplotlib.pyplot as plt
import random


def _calc_dis(p1,p2):
    return np.sqrt(np.sum((p1-p2)**2))

# Kuang 2012, Figure 1: Average ocular drift speeds (D), and lengths (E)
drift_size=0.17 # degree

unit_time=10 # ms
sigma=0.33

target_pos=np.array([0,0])

# Zhang 2010
dwell_time_minimum=800
tt=np.array([1.23,1.72,2.15,2.64,3.07]) 



for j in range(len(tt)):
	target_size=tt[j]# degrees

	dis_to_target=target_size
	dwell_time=[]
	
	for i in range(1000): 

		# choose a random point at the first point entering the target region
		while dis_to_target>=target_size/2:
			start=np.random.uniform(low=-target_size/2,high=target_size/2,size=(2,))
			dis_to_target=_calc_dis(start,target_pos)


		x=start[0]
		y=start[1]

		trial_time=0

		time_count=0
		tiral_count=0

		new_pos=start

		while time_count<dwell_time_minimum and tiral_count<110:


			'''
			x+=random.choice([-drift_size,drift_size])
			y+=random.choice([-drift_size,drift_size])

			new_pos=np.array([x,y])
			'''

			new_pos=np.random.normal(target_pos,sigma,(2,))
			dis_to_target=_calc_dis(new_pos,target_pos)
			time_count+=unit_time
			trial_time+=unit_time
			tiral_count+=1

			if dis_to_target>=target_size/2:

				while dis_to_target>=target_size/2:
					start=np.random.uniform(low=-target_size/2,high=target_size/2,size=(2,))
					dis_to_target=_calc_dis(start,target_pos)

				x=start[0]
				y=start[1]
				time_count=0

		print(f'all_time is {trial_time} ms')

		dwell_time.append(trial_time)
	plt.subplot(2,3,j+1)
	plt.hist(dwell_time)
	plt.title((np.mean(dwell_time)))

plt.show()


