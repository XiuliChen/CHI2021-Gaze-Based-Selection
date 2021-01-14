
import os
import csv

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback



from envs.gaze import Gaze
from envs.utils import calc_dis, moving_average,plot_results2
from numpy import genfromtxt



def main(fitts_W,fitts_D,ocular_std,swapping_std,run,timesteps,logs_folder):
    # Create log dir
    lc_dir = f'./{logs_folder}/w{fitts_W}d{fitts_D}ocular{ocular_std}swapping{swapping_std}/'
    log_dir = f'{lc_dir}/run{run}/'
    os.makedirs(log_dir, exist_ok=True)

    # Instantiate the env
    env = Gaze(fitts_W = fitts_W, fitts_D=fitts_D, ocular_std=ocular_std, swapping_std=swapping_std)
    env = Monitor(env, log_dir)

    # Train the agent
    model = PPO('MlpPolicy', env, verbose=0, clip_range=0.15)
    model.learn(total_timesteps=int(timesteps))
    # save the model
    model.save(f'{log_dir}savedmodel/model_ppo')

    # plot learning curve
    plot_results2(log_dir)

    plt.savefig(f'{lc_dir}learning_curve{run}.png')
    plt.close('all') 

    ###########################################################################
    # Record Behaviour of the trained policy
    ###########################################################################
    # save the step data

    # Test the trained agent
    n_eps=5000
    number_of_saccades=np.ndarray(shape=(n_eps,1), dtype=np.float32)
    eps=0
    while eps<n_eps:                
        done=False
        step=0
        obs= env.reset()
        while not done:
            step+=1
            action, _ = model.predict(obs,deterministic = True)
            obs, reward, done, info = env.step(action)
            if done:
                number_of_saccades[eps]=step
                eps+=1
                break
    
    np.savetxt( f'{log_dir}num_saccades.csv', number_of_saccades, delimiter=',')


            
if __name__ == '__main__':
    ###########################################################################
    #TRAINING
    ###########################################################################
    # the target distance and width from Zhang2010 and Schutez2019
    # In the model, the start position is [0,0]
    # the display is x=[-1,1], and y=[-1,1]
    # 25.78 degree is converted to 0.5 
    # and other distance and width is converted proportionally.

    #d_zhang=np.array([25.78,11.68,6.16])
    d_zhang=np.array([6.16])
    w_zhang=np.array([1.23, 1.73, 2.16, 2.65, 3.08])

    w_schuetz=np.array([5,4,3,2,1.5,1])
    d_schuetz=np.array([5,10])

    unit=0.5/11.68

    w_zhang=np.round(w_zhang*unit,2)
    d_zhang=np.round(d_zhang*unit,2)

    w_schuetz=np.round(w_schuetz*unit,2)
    d_schuetz=np.round(d_schuetz*unit,2)

    ocular_std=0.07
    swapping_std=0.09

    timesteps = 3e6
    logs_folder=f'saved_models/logs_{ocular_std}_{swapping_std}'
    ###########################################################################
    for paper in ['schuetz','zhang']:
        if paper=='schuetz':
            w=w_schuetz
            d=d_schuetz
        else:
            w=w_zhang
            d=d_zhang   

        for fitts_W in w:
            for fitts_D in d:        
                for run in [5]:
                    main(fitts_W,fitts_D,ocular_std,swapping_std,run,timesteps,logs_folder)

