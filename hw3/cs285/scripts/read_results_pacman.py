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
            elif v.tag == 'Train_BestReturn':
                Y.append(v.simple_value)
    return X, Y#, Z

if __name__ == '__main__':
    import glob

    logdirs = ['/home/rasp/Scrivania/hw3/cs285/data/pacman/events*', '/home/rasp/Scrivania/hw3/cs285/data/pacman_qq/events*']
    titles = ["Vanilla Q-learning", "Double Q-Learning"]
    for i, logdir in enumerate(logdirs):
	    eventfile = glob.glob(logdir)[0]
	    X, Y = get_section_results(eventfile)
	    X, Y = X[:35], Y[:35]
	    plt.plot(X, Y, label = titles[i])

    plt.xlabel("Iteration number")
    plt.ylabel("Train_BestReturn")
    plt.legend()
    plt.savefig("lunar_lander_q.png")
