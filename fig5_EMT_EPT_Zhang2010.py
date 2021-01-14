
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


latency=260


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
d_schuetz=np.array([10,5])

unit=0.5/11.68

w_zhang=np.round(w_zhang*unit,2)
d_zhang=np.round(d_zhang*unit,2)

w_schuetz=np.round(w_schuetz*unit,2)
d_schuetz=np.round(d_schuetz*unit,2)

###########################################################################
for paper in ['zhang']:
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
        fig=plt.figure(figsize=(8,6.5))
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
            number_of_saccades=np.ndarray(shape=(n_eps,1), dtype=np.float32)
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
                    amp=move_dis/(unit)
                    
                    movement_time+=saccade_time(amp)+latency
                    if done:
                        number_of_saccades[eps]=step
                        movement_time_all[eps]=movement_time
                        eps+=1
                        break

            EMT_mean.append(np.round(np.mean(movement_time_all),2))
            EMT_std.append(np.round(np.std(movement_time_all),2))


        


        plt.plot(target_size, EMT_mean, 'k<:', markersize=10,label= f'Eye movement time (model)')
      
        extra=np.array([1100,1067,881,807,800]) # check Jitter_Zhang.py
        plt.plot(target_size, EMT_mean+extra,'ro--',markersize=10,label='Selection time (model)')
        
        #plot data
        print('plot data given fromt the first author of Zhang et al., 2010')
        print(np.array(EMT_mean))


        EMT=np.array([549,452,374,314,324])
        EPT=np.array([1624,1421,1292,1202,1200])

        print(RMSE(np.array(EMT_mean),EMT))
        print(RMSE(np.array(EMT_mean+extra), EPT))

        width=0.3
        ax=fig.gca()
        ax.bar(target_size, EMT, width, color='y', edgecolor='black', hatch="/",label='Eye movement time (data)')
        plt.bar(target_size, EPT-EMT, width, 
        	bottom=EMT, color=colors[1],label='Selection time (data)')

        plt.xlabel(f'Target size({chr(176)})', fontsize=15)
        plt.ylabel('Time(ms)', fontsize=15)
        plt.xticks(target_size, fontsize=12)
        plt.legend(title=r' $\rho_{ocular}$' f'={ocular_std},' r' $\rho_{spatial}$' f'={swapping_std}',fontsize=14)

        plt.ylim([200,2000])
        plt.yticks(np.arange(0,2000,250), fontsize=12)


        dwell=[1075,969,918,890,876]
        for i in range(5):
        	ax.text(target_size[i]-0.1,800,f'{dwell[i]}',rotation=90,fontsize=18)

    plt.savefig(f'figures/Zhang2010_EMT_EPT.pdf')
    plt.show()





 