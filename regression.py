import numpy as np
import matplotlib.pyplot as plt
import random
def f(x):
    return np.exp(-1/2*np.square(x-2))*np.sin(((3*np.pi)/2)*x)-np.exp(-1/2*np.square(x+2))*np.cos(np.pi*x)

def plotFunction():
    # 100 linearly spaced numbers
    x = np.linspace(-5, 5, 100)
    y = f(x)
    # setting the axes at the centre
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # plot the function
    plt.plot(x, y, 'r')

    # show the plot
    plt.show()

def scatterPlot(x,y):
    plt.scatter(x, y)
    plt.show()

def generateTraining(N):
    x = np.random.uniform(low=-5, high=5, size=(N,))
    y = f(x)
    return x,y
if __name__ == '__main__':
    x,y = generateTraining(30)
    scatterPlot(x,y)

