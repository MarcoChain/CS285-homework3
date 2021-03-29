import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(color_codes=True)

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
    return X, Y
def average(vect, window_size = 10):
	res = np.zeros(len(vect))
	res[:window_size] = vect[:window_size]
	for i in range(window_size, len(vect)):
		res[i] = vect[i-window_size:i].mean()
	return res
	
if __name__ == '__main__':
    import glob

    logdir = '/home/rasp/Scrivania/hw3/cs285/data/HalfCheetah/events*'
    eventfile = glob.glob(logdir)[0]
    X, Y = get_section_results(eventfile)
    plt.plot(Y)
    plt.ylabel("Eval_average_return")
    plt.xlabel("Iteration number")
    #plt.legend()
    plt.savefig("ex_half.png")
    
    
