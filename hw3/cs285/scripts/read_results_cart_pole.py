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
    return X, Y

if __name__ == '__main__':
    import glob

    logdirs = ['/home/rasp/Scrivania/hw3/cs285/data/cart_pole_1_1/events*', '/home/rasp/Scrivania/hw3/cs285/data/cart_pole_1_100/events*', '/home/rasp/Scrivania/hw3/cs285/data/cart_pole_100_1/events*', '/home/rasp/Scrivania/hw3/cs285/data/cart_pole_10_10/events*']
    titles = ["ntu_1_ngsput_1", "ntu_1_ngsput_100", "ntu_100_ngsput_1", "ntu_10_ngsput_10"]

    for title, logdir in zip(titles, logdirs):
	    eventfile = glob.glob(logdir)[0]
	    X, Y = get_section_results(eventfile)
	    plt.plot(np.linspace(0,100, 10), Y, label = title)
    plt.xlabel("Iteration number")
    plt.ylabel("Train_Average_Return")
    plt.legend()
    plt.savefig("cart_pole.png")
