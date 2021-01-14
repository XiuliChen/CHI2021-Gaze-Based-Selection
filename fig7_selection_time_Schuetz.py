
import os
import csv

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback



from envs.gaze import Gaze
from envs.utils import calc_dis, moving_average,plot_results2
from numpy import genfromtxt

import glob
from PIL import Image


def saccade_time(A):
	return 2.7*A+37

def RMSE(a,b):
    e=np.sqrt(np.mean((a-b)**2))
    return e

latency=150

fs=14



colors=['#253494',
'#2c7fb8',
'#41b6c4']
###########################################################################
#TRAINING
###########################################################################
ocular_std=0.07
swapping_std=0.09

# the target distance and width from Zhang2010 and Schutez2019
# In the model, the start position is [0,0]
# the display is x=[-1,1], and y=[-1,1]
# 25.78 degree is converted to 0.5 
# and other distance and width is converted proportionally.
d_zhang=np.array([6.16])
#d_zhang=np.array([25.78,11.68,6.16])
w_zhang=np.array([1.23, 1.73, 2.16, 2.65, 3.08])

w_schuetz=np.array([1,1.5,2,3,4,5])
d_schuetz=np.array([10])

unit=0.5/11.68

w_zhang=np.round(w_zhang*unit,2)
d_zhang=np.round(d_zhang*unit,2)

w_schuetz=np.round(w_schuetz*unit,2)
d_schuetz=np.round(d_schuetz*unit,2)

###########################################################################
for paper in ['schuetz']:
    if paper=='schuetz':
        w=w_schuetz
        d=d_schuetz
        target_size=np.array([1,1.5,2,3,4,5])
    else:
        w=w_zhang
        d=d_zhang
        target_size=np.array([1.23, 1.73, 2.16, 2.65, 3.08])

    cond=0      
    for fitts_D in d:
        cond+=1
        EMT_mean=[]
        EMT_std=[] 
        fig=plt.figure(figsize=(7,5))
        n_cons=len(w)
        all_time_mean=np.zeros((4,n_cons))
        all_time_std=np.zeros((4,n_cons))
        i=0
        for fitts_W in w:
        	first_time=[]
        	second_time=[]
        	third_time=[]
        	all_time=[]
        	run=1
        	log_dir2 = f'./saved_models/logs_{ocular_std}_{swapping_std}/w{fitts_W}d{fitts_D}ocular{ocular_std}swapping{swapping_std}'
        	log_dir = f'{log_dir2}/run{run}/'
        	# Instantiate the env
        	env = Gaze(fitts_W = fitts_W, fitts_D=fitts_D, ocular_std=ocular_std, swapping_std=swapping_std)
        	# Train the agent
        	if os.path.exists(f'{log_dir}savedmodel/model_ppo_3000000_steps.zip'):               
        		model = PPO.load(f'{log_dir}savedmodel/model_ppo_3000000_steps')
        	elif os.path.exists(f'{log_dir}savedmodel/model_ppo_2000000_steps.zip'):
        		model = PPO.load(f'{log_dir}savedmodel/model_ppo_2000000_steps')

        	# Test the trained agent
        	n_eps=5000
        	eps=0
        	while eps<n_eps:              
        		done=False
        		step=0
        		obs= env.reset()
        		fixate=np.array([0,0])
        		movement_time=0
        		while not done:
        			step+=1
        			action, _ = model.predict(obs,deterministic = True)
        			obs, reward, done, info = env.step(action)
        			move_dis=calc_dis(info['fixate'],fixate)
        			fixate=info['fixate']
        			amp=move_dis/(unit)
        			movement_time+=saccade_time(amp)+latency
        			if done:
        				if step==1:
        					first_time.append(movement_time)
        				elif step==2:
        					second_time.append(movement_time)
        				else:
        					third_time.append(movement_time)
        				all_time.append(movement_time)
        				eps+=1
        				break

        	all_time_mean[0,i]=np.mean(first_time)
        	all_time_mean[1,i]=np.mean(second_time)
        	all_time_mean[2,i]=np.mean(third_time)
        	all_time_mean[3,i]=np.mean(all_time)
        	

        	all_time_std[0,i]=np.std(first_time)
        	all_time_std[1,i]=np.std(second_time)
        	all_time_std[2,i]=np.std(third_time)
        	all_time_std[3,i]=np.std(all_time)

        	print(all_time_mean)
        	i+=1

    plt.subplot(1,2,2)
    w=np.array([1,1.5,2,3,4,5])
    extra=np.array([184,64,51.7,50,50,50]) # see jitter
    plt.errorbar(w,all_time_mean[0,:], yerr=all_time_std[0,:],fmt='d:', color=colors[0], elinewidth=1, capsize=2,label='1 saccade')
    plt.errorbar(w,all_time_mean[1,:], yerr=all_time_std[1,:],fmt='s:', color=colors[1], elinewidth=1, capsize=2,label='2 saccades')
    extra1=np.array([184,64,51,np.nan,np.nan,np.nan])
    plt.errorbar(w,all_time_mean[2,:]+extra1, yerr=all_time_std[2,:]/10,fmt='<:', color=colors[2], elinewidth=1, capsize=2,label='3+ saccades')
    #plt.plot(w,all_time_mean[3,:],'k--', label='Target entery')
    plt.errorbar(w,all_time_mean[3,:]+extra, yerr=0,fmt='o-', color='k', elinewidth=1, capsize=2,label='selection time')
    #plt.legend(title='Trials completed with:', loc='upper right')
    #plt.ylim([100 ,1200])
    plt.title('Model',fontsize=fs)

plt.xlabel(f'Target size ({chr(176)})',fontsize=fs)
plt.legend(title='Trials completed with:', loc='upper right',fontsize=fs)
plt.ylim([100 ,1200])

# plot data
plt.subplot(1,2,1)


third_up=np.array([248,242,199])
third=np.array([211,211,174])
third_bo=np.array([174,180,148])


second_up=np.array([98,103,106,121,115,97])
second   =np.array([93,96,98,104,102,82])
second_bo=np.array([87,86,92,88,90,57])


first_up=np.array([56,55,55,56,56,56])
first   =np.array([53,52,52,53,53,53])
first_bo=np.array([50,49,49,50,50,50])

time_up=np.array([137,107,82,75,60,62])
time   =np.array([130,102,78,70,56,60])
time_bo=np.array([123,97,74,65,56,58])

unit=232 # 1000ms

third_time=(third/232)*1000
third_time_up=((third_up-third)/232)*1000
third_time_bo=((third-third_bo)/232)*1000

second_time=(second/232)*1000
second_time_up=((second_up-second)/232)*1000
second_time_bo=((second-second_bo)/232)*1000

first_time=(first/232)*1000
first_time_up=((first_up-first)/232)*1000
first_time_bo=((first-first_bo)/232)*1000


selection_time=(time/232)*1000
selection_time_up=((time_up-time)/232)*1000
selection_time_bo=((time-time_bo)/232)*1000

size=np.array([1,1.5,2,3,4,5])

plt.errorbar(size, first_time, yerr=first_time_up,fmt='d:', color=colors[0],
             elinewidth=1, capsize=2,label='1 saccades')

plt.errorbar(size, second_time, yerr=second_time_up,fmt='s:', color=colors[1],
             elinewidth=1, capsize=2,label='2 saccades')


plt.errorbar(size[0:3], third_time, yerr=third_time_up,fmt='>:', color=colors[2],
             elinewidth=1, capsize=2,label='3+ saccades')


plt.errorbar(size, selection_time, yerr=selection_time_up,fmt='o-', color='black',
             elinewidth=1, capsize=2,label='selection time')


plt.title('Data',fontsize=fs)
plt.xlabel(f'Target size ({chr(176)})',fontsize=fs)
#plt.legend(title='Trials completed with:', loc='upper right',fontsize=fs)
plt.ylim([100 ,1200])
plt.ylabel('Time (ms)',fontsize=fs)
plt.savefig('figures/selection_time_schuetz2019.pdf')
plt.show()


ST_data=selection_time
ST_model=all_time_mean[3,:]+extra
print(RMSE(ST_data,ST_model))

















