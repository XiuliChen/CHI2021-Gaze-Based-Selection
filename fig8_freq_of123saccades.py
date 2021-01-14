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



colors=['#1b9e77',
'#d95f02',
'#7570b3',
'#225ea8',
'#0c2c84']
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
d_zhang=np.array([11.68,6.16])
#d_zhang=np.array([25.78,11.68,6.16])
w_zhang=np.array([1.23, 1.73, 2.16, 2.65, 3.08])

w_schuetz=np.array([1,1.5,2,3,4,5])
d_schuetz=np.array([5,10])

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
    plt.figure(figsize=(8,4))    
    for fitts_D in d:
        cond+=1
        saccade_mean=[]
        saccade_std=[]

        ff=np.zeros((6,4))
        i=0
        for fitts_W in w:
            run=1
            # Create log dir
            log_dir2 = f'./saved_models/logs_{ocular_std}_{swapping_std}/w{fitts_W}d{fitts_D}ocular{ocular_std}swapping{swapping_std}'
            log_dir = f'{log_dir2}/run{run}/'
            
            # Instantiate the env
            env = Gaze(fitts_W = fitts_W, 
                fitts_D=fitts_D, 
                ocular_std=ocular_std, 
                swapping_std=swapping_std)


            # Custom MLP policy of two layers of size 32 each with tanh activation function
            #policy_kwargs = dict(net_arch=[128, 128])
            #policy_kwargs=policy_kwargs
           
            # Train the agent
            

            if os.path.exists(f'{log_dir}savedmodel/model_ppo_3000000_steps.zip'):               
                model = PPO.load(f'{log_dir}savedmodel/model_ppo_3000000_steps')
            elif os.path.exists(f'{log_dir}savedmodel/model_ppo_2000000_steps.zip'):
                model = PPO.load(f'{log_dir}savedmodel/model_ppo_2000000_steps')


            env = Gaze(fitts_W = fitts_W, 
                fitts_D=fitts_D, 
                ocular_std=ocular_std, 
                swapping_std=swapping_std)

            # Test the trained agent
            n_eps=5000
            number_of_saccades=[]
            movement_time_all=np.ndarray(shape=(n_eps,1), dtype=np.float32)
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
                    movement_time+=37+2.7*move_dis
                    if done:
                        number_of_saccades.append(step)
                        eps+=1
            a,b=np.unique(number_of_saccades, return_counts=True)

            ff[i,a-1]=b
            i+=1

        dataset=ff/n_eps
        print(dataset)

       	ind=np.array([1,1.5,2,3,4,5])
       	plt.subplot(1,2,cond)
       	width = 0.2
       	p1 = plt.bar(ind, dataset[:,0], width, color=colors[0],label='1 fixation')
       	p2 = plt.bar(ind, dataset[:,1], width, bottom=dataset[:,0], color=colors[1],label='2 fixations')
       	p3 = plt.bar(ind, dataset[:,2], width, bottom=np.array(dataset[:,0])+np.array(dataset[:,1]), color=colors[2],label='3 fixations')
       	p4 = plt.bar(ind, dataset[:,3], width,
		             bottom=np.array(dataset[:,0])+np.array(dataset[:,1])+np.array(dataset[:,2]),
		             color=colors[2])
        fs=14
        plt.xlabel(f'Target size{chr(176)}',fontsize=fs)
        plt.ylabel('Proportion of trials',fontsize=fs)
        if cond==1:
            plt.title(f'Distance=5{chr(176)}',fontsize=fs)
        else:
            plt.title(f'Distance=10{chr(176)}',fontsize=fs)


       	plt.legend(title='Trials completed with:', loc='lower right')
plt.savefig('figures/frequency123_schuetz2019.pdf')
plt.show()

	


