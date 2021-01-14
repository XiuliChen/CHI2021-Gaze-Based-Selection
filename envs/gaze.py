
# By Xiuli Chen 04/Sep/2020
# chsh -s /bin/bash


import numpy as np
import gym
from gym import spaces
import math
import matplotlib.pyplot as plt
import itertools


###########################################################################
# some tools used in the class Gaze defined below
###########################################################################

def _calc_dis(p, q):
    return np.sqrt(np.sum((p - q)**2))

def get_new_target(D):
    '''
    generate a target at a random angle, distance D away.
    '''
    angle=np.random.uniform(0,math.pi*2) 
    x_target=math.cos(angle)*D
    y_target=math.sin(angle)*D
    return np.array([x_target,y_target])
###########################################################################


class Gaze(gym.Env):
    '''
    Description:
            The agent moves the eye to the target on the dispaly. 
            The agent's vision has spactial swapping noise that is a funtion 
            of the eccentricity

    States: the target position (type: Box(2, ));
            [-1,-1] top-left; [1,1] bottom-right 

    Actions: the fixation position (type: Box(2,));
            [-1,-1] top-left; [1,1] bottom-right 

    Observation: the estimate of where the target is based on one obs
            (type: Box(2, ));
            [-1,-1] top-left; [1,1] bottom-right 

    Belief: the estimate of where the target is based on all obs
            (type: Box(2, ));
            [-1,-1] top-left; [1,1] bottom-right 


    Reward:
            Reward of 0 is awarded if the eye reach the target.
            reward of -1 is awared if not


    Episode Termination:
            the eye reaches the target (within self.fitts_W/2)
            or reach the maximum steps

    '''

    def __init__(self,fitts_W, fitts_D,ocular_std, swapping_std):
        # task setting
        self.fitts_W = fitts_W
        self.fitts_D= fitts_D

        # agent ocular motor noise and visual spatial noise
        self.ocular_std=ocular_std
        self.swapping_std=swapping_std
         

        ## state, action and observation space
        # where to fixate
        self.action_space = spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float64)
        # the target location
        self.state_space = spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float64)

        #observation
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float64)
        self.belief_space = spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float64)

        self.max_fixation=500

         
    def reset(self):
        # choose a random one target location
        self.state = get_new_target(self.fitts_D)

        # As in the experiments the participants starting with a 
        # fixation at the starting point, the agent start with the
        # first fixation at the center
        self.fixate=np.array([0,0])
        self.n_fixation=1

        # the first obs
        self.obs,self.obs_uncertainty=self._get_obs()

        # the initial belief of the target location is based on
        # the first obs
        self.belief,self.belief_uncertainty=self.obs, self.obs_uncertainty


        return self.belief

    def step(self, action):
        # execute the chosen action given the ocular motor noise
        move_dis=_calc_dis(self.fixate,action)
        ocular_noise=np.random.normal(0, self.ocular_std*move_dis, action.shape)
        self.fixate= action + ocular_noise
        self.fixate=np.clip(self.fixate,-1,1)

        others={'n_fixation':self.n_fixation,
                'target': self.state, 
                'belief': self.belief,
                'aim': action,
                'fixate': self.fixate}

        self.n_fixation+=1

        # check if the eye is within the target region
        dis_to_target=_calc_dis(self.state, self.fixate)
        
        if  dis_to_target < self.fitts_W/2:
            done = True
            reward = 0
        else:
            done = False
            reward = -1 
            # has not reached the target, get new obs at the new fixation location
            self.obs,self.obs_uncertainty=self._get_obs()
            self.belief,self.belief_uncertainty=self._get_belief()

        if self.n_fixation>self.max_fixation:
            done=True

        return self.belief, reward, done, others


    def _get_obs(self):
        eccentricity=_calc_dis(self.state,self.fixate)
        obs_uncertainty=eccentricity
        spatial_noise=np.random.normal(0, self.swapping_std*eccentricity, self.state.shape)
        obs=self.state + spatial_noise
        obs=np.clip(obs,-1,1)
        
        return obs,obs_uncertainty

    def _get_belief(self):
        z1,sigma1=self.obs,self.obs_uncertainty
        z2,sigma2=self.belief,self.belief_uncertainty

        w1=sigma2**2/(sigma1**2+sigma2**2)
        w2=sigma1**2/(sigma1**2+sigma2**2)

        belief=w1*z1+w2*z2
        belief_uncertainty=np.sqrt( (sigma1**2 * sigma2**2)/(sigma1**2 + sigma2**2))

        return belief, belief_uncertainty

    

    def render(self):
        pass





