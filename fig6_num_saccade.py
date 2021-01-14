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

def RMSE(a,b):
    e=np.sqrt(np.mean((a-b)**2))
    return e

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
d_schuetz=np.array([10,5])

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
    plt.figure(figsize=(4.5,4.5))

    cond=0      
    for fitts_D in d:
        cond+=1
        saccade_mean=[]
        saccade_std=[] 
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
                    movement_time+=37+2.7*move_dis
                    if done:
                        number_of_saccades[eps]=step
                        movement_time_all[eps]=movement_time
                        eps+=1
                        break
            

            #np.savetxt( f'{log_dir}num_saccades.csv', number_of_saccades, delimiter=',')
            #np.savetxt( f'{log_dir}movement_time.csv', movement_time_all, delimiter=',') 
            
            saccade_mean.append(np.round(np.mean(number_of_saccades),2))
            saccade_std.append(np.round(np.std(number_of_saccades),2))


        


        if cond==1:
            plt.plot(target_size, saccade_mean, 'd:', label= f'Model (D={np.round(fitts_D/unit)} {chr(176)})')
        else:
            plt.plot(target_size, saccade_mean, 's-.', label= f'Model (D={np.round(fitts_D/unit)} {chr(176)})')

        #plt.errorbar(param_values, saccade_mean, yerr=saccade_std,fmt='o-', label=f'ocular noise={ocular_std}')

        if cond==1:
            tmp=saccade_mean
        else:
            saccade_mean_poolled=(np.array(tmp)+np.array(saccade_mean))/2
            plt.plot(target_size, saccade_mean_poolled, 'k>--', label= f'Model (D=10 {chr(176)} and 5 {chr(176)} pooled)')

           #plot data

    n=np.array([182,171,128,48,31,20])
    up=np.array([226,205,162,70,44,30])
    bo=np.array([136,137,93,27,18,9])

    num_fix=1+n/349
    err1=(up-n)/349
    err2=(n-bo)/349



    plt.errorbar(target_size, num_fix, yerr=err1,fmt='ko-', color='black',
                 ecolor='black',  capsize=2,label=f'Data (D=10 {chr(176)} and 5 {chr(176)} pooled)')


    plt.xlabel('Target size', fontsize=13)
    plt.ylabel('The number of saccade per trial', fontsize=13)
    plt.ylim(0.9,2.45)
    plt.xticks(target_size)
    plt.legend(title=r' $\rho_{ocular}$' f'={ocular_std},' r' $\rho_{spatial}$' f'={swapping_std}',fontsize=11)
    




    print(RMSE(saccade_mean_poolled,num_fix))
    plt.savefig(f'figures/num_saccades_schuetz2019.pdf')
    plt.show()



