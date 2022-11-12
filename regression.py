import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.exp(-1/2*np.square(x-2))*np.sin(((3*np.pi)/2)*x)-np.exp(-1/2*np.square(x+2))*np.cos(np.pi*x)


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

