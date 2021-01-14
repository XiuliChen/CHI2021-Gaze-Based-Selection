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
from main import main


###########################################################################
#TRAINING
###########################################################################


# the target distance and width from Schutez2019

w_schuetz=np.array([1,1.5,2,3,4,5])
d_schuetz=np.array([10,5])

unit=0.5/11.68

w_schuetz=np.round(w_schuetz*unit,2)
d_schuetz=np.round(d_schuetz*unit,2)

###########################################################################

w=w_schuetz
d=d_schuetz

fitts_W=w[1] #W=1.5 deg
fitts_D=d[0] #D=10 deg

timesteps = 1e5
logs_folder='sensitivity_test_saved_models'

param_values=np.array([0.15,0.125,0.1,0.075,0.05,0.025,0.001])
run=7

for ocular_std in param_values:
    for swapping_std in param_values:
        main(fitts_W,fitts_D,ocular_std,swapping_std,run,timesteps,logs_folder)
