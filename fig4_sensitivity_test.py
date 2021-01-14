
import os
import os.path
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
from envs.utils import calc_dis
from numpy import genfromtxt

import glob
from PIL import Image

colors=['#b10026',
'#e31a1c',
'#fc4e2a',
'#fd8d3c',
'#feb24c',
'#fed976',
'#ffffb2']


colors=['#a6cee3',
'#1f78b4',
'#b2df8a',
'#33a02c',
'#fb9a99',
'#e31a1c',
'#fdbf6f',
]

# the target distance and width from Zhang2010 and Schutez2019
# In the model, the start position is [0,0]
# the display is x=[-1,1], and y=[-1,1]
# 25.78 degree is converted to 0.5 
# and other distance and width is converted proportionally.

#d_zhang=np.array([25.78,11.68,6.16])
d_zhang=np.array([6.16])
w_zhang=np.array([1.23, 1.73, 2.16, 2.65, 3.08])

w_schuetz=np.array([1,1.5,2,3,4,5])
d_schuetz=np.array([10,5])

unit=0.5/11.68

w_zhang=np.round(w_zhang*unit,2)
d_zhang=np.round(d_zhang*unit,2)

w_schuetz=np.round(w_schuetz*unit,2)
d_schuetz=np.round(d_schuetz*unit,2)

plt.figure(figsize=(6,6))
###########################################################################
for paper in ['schuetz']:
    if paper=='schuetz':
        w=w_schuetz
        d=d_schuetz
        fitts_W=w[1] #W=1.5 deg
        fitts_D=d[0] #D=10 deg
    else:
        w=w_zhang
        d=d_zhang 
        fitts_W=w[1] #W=1.73 deg
        fitts_D=d[1] #D=6.16 deg 

    param_values=np.array([0.15,0.125,0.1,0.075,0.05,0.025,0.001])

    count=-1
    #
    for ocular_std in [0.15,0.125,0.1,0.075,0.05,0.025,0.0001]:
        print(f'ocular_std{ocular_std}')
        saccade_mean=[]
        saccade_std=[]       
        count+=1
        for swapping_std in [0.15,0.125,0.1,0.075,0.05,0.025,0.0001]:
            print(f'swapping_std{swapping_std}')

            run=5
            log_dir = f'sensitivity_test_saved_models/w{fitts_W}d{fitts_D}ocular{ocular_std}swapping{swapping_std}/run{run}/'
            print(log_dir)
            model = PPO.load(f'{log_dir}savedmodel/model_ppo')
            env = Gaze(fitts_W = fitts_W, 
                fitts_D=fitts_D, 
                ocular_std=ocular_std, 
                swapping_std=swapping_std)


            my_data = genfromtxt(f'{log_dir}num_saccades.csv', delimiter=',')
            
            saccade_mean.append(np.mean(my_data))
            saccade_std.append(np.std(my_data))
            print(np.mean(my_data))

        plt.plot(param_values, saccade_mean, 'o:',color=colors[count], label= r' $\rho_{ocular}$' f'={ocular_std}')
        #plt.errorbar(param_values, saccade_mean, yerr=saccade_std,fmt='o-', label=f'ocular noise={ocular_std}')


        plt.xlabel(r'Visual spatial noise: $\rho_{spatial}$',fontsize=14)
        plt.ylabel('The number of saccade per trial',fontsize=14)
        plt.ylim(0.2,2.3)
        plt.xticks(param_values)
        plt.legend(title=f'Task:D=10 {chr(176)}, W=1.5{chr(176)} \n Ocular motor noise:',fontsize=11)
        




    plt.savefig(f'figures/sensitivity_test.pdf')
    plt.show()



