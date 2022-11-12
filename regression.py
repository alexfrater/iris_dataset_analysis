
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

def f(x):
    return np.exp(-1/2*np.square(x-2))*np.sin(((3*np.pi)/2)*x)-np.exp(-1/2*np.square(x+2))*np.cos(np.pi*x)

def plotFunction(p):
    # 100 linearly spaced numbers
    x = np.linspace(-5, 5, 100)
    y = polyEqn(x,p)
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


def scatterPlot(x,y):
    plt.scatter(x, y)
    plt.show()

def generateSetRandom(N):
    x = np.random.uniform(low=-5, high=5, size=(N,))
    y = f(x)
    return x,y

def generateSetArranged(N):
    x = np.arange(5)
    y = f(x)
    return x,y

def normal(X,Y):
    return np.dot(np.dot(X.T,Y), la.inv(np.dot(X.T,X)))

def x_poly(x,n):
    X=x
    for i in range(n-1):
        X = np.c_[X,np.power(x, i+2)]
    return X
def polyEqn(x,m):
    y = 0

    for i in range(1,len(m)):
        y = y + m[i]*np.power(x,i)
    return  y

if __name__ == '__main__':


    x_training,y_training = generateSetRandom(200)
    #scatterPlot(x_training,y_training)
    X = x_poly(x_training,10)

    m = normal(X,y_training)

    plotFunction(m)
    scatterPlot(x_training,y_training)
    plt.show()





