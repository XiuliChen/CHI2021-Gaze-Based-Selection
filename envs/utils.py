from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
import matplotlib.pyplot as plt
import numpy as np
import math

# for plotting the learning curve
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

# for plotting the learning curve
def plot_results2(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=100)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')

    
# some tool functions
def calc_dis(p,q):
    #calculate the Euclidean distance between points p and q 
    return np.sqrt(np.sum((p-q)**2))

###########################################################

def get_new_target(D):
    '''
    generate a target at a random angle, distance D away.
    '''
    angle=np.random.uniform(0,math.pi*2) 
    x_target=math.cos(angle)*D
    y_target=math.sin(angle)*D
    return np.array([x_target,y_target])
###########################################################





if __name__=="__main__":
    import matplotlib.pyplot as plt
    # unit testing
    p=np.array([0,0])
    q=np.array([1,1])
    dis=calc_dis(p,q)
    print(dis)

    D=0.5
    for i in range(100):
        target_pos=get_new_target(D)
        plt.plot(target_pos[0],target_pos[1],'ro')
 

    plt.figure()
    
    mode=1 # eye
    current_pos=np.array([0,0])
    actual_pos=np.array([0,0.7])
    amp=calc_dis(current_pos,actual_pos)*20
    time_step=20

    




    




    
    
