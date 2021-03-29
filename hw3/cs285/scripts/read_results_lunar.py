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
            #elif v.tag == 'Train_BestReturn':
                #Z.append(v.simple_value)
    return X, Y#, Z

if __name__ == '__main__':
    import glob

    #logdirs = ['/home/rasp/Scrivania/hw3/cs285/data/luna_qq_1/events*', '/home/rasp/Scrivania/hw3/cs285/data/luna_qq_2/events*', '/home/rasp/Scrivania/hw3/cs285/data/luna_qq_3/events*']
    #logdirs = ['/home/rasp/Scrivania/hw3/cs285/data/luna_q_1/events*', '/home/rasp/Scrivania/hw3/cs285/data/luna_q_2/events*', '/home/rasp/Scrivania/hw3/cs285/data/luna_q_3/events*']
    logdirs = ['/home/rasp/Scrivania/hw3/cs285/data/LunarLander_batch_size_64/events*', '/home/rasp/Scrivania/hw3/cs285/data/LunarLander_batch_size_128/events*', '/home/rasp/Scrivania/hw3/cs285/data/LunarLander_batch_size_256/events*']
    titles = ["batch_size 64", "batch_size 128", "batch_size 256"]
    for title, logdir in zip(titles, logdirs):
	    eventfile = glob.glob(logdir)[0]
	    X, Y = get_section_results(eventfile)
	    #X, Y = X[:36], Y[:36]
	    X = X[:49]
	    #plt.plot(X, Y, label = f"Seed_{i+1}")
	    plt.plot(X, Y, label = title)
	    #plt.plot(X, Z, label = "Train_BestReturn")
    plt.xlabel("Iteration number")
    plt.ylabel("Train_Average_Return")
    plt.legend()
    plt.savefig("lunar_lander_batch_size.png")
