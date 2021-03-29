import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(color_codes=True)

def get_section_results(file):
    """4
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    Z = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
            elif v.tag == 'Train_BestReturn':
                Z.append(v.simple_value)
    return X, Y, Z

if __name__ == '__main__':
    import glob

    logdir = '/home/rasp/Scrivania/hw3/cs285/data/pacman/events*'
    eventfile = glob.glob(logdir)[0]
    X, Y, Z = get_section_results(eventfile)
    #for i, (x, y, z) in enumerate(zip(X, Y, Z)):
        #print('Iteration {:d} | Train steps: {:d} | Return: {} | Max Return: {}'.format(i, int(x), y, z ))
    plt.plot(np.log(X[: -2]), Y[: -1], label = "Train_AverageReturn")
    plt.plot(np.log(X[:-2]), Z, label = "Train_BestReturn")
    plt.xlabel("Iteration number")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig("pacman_results_log.png")
