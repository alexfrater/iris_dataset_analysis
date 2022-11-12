
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la


class plotter:
    fig = None
    ax = None
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.spines['left'].set_position('center')
        self.ax.spines['bottom'].set_position('zero')
        self.ax.spines['right'].set_color('none')
        self.ax.spines['top'].set_color('none')
        self.ax.xaxis.set_ticks_position('bottom')
        self.ax.yaxis.set_ticks_position('left')
        self.ax.set_xlim(xmin=-5, xmax=5)
        self.ax.set_ylim(ymin=-1.5, ymax=1.5)

    def plotXY(self,x, y, color):
        # plot the function
        self.ax.plot(x, y, color)

    def scatterPlot(self,x,y):
        self.ax.scatter(x, y)


# body of the constructor
def f(x):
    return np.exp(-1/2*np.square(x-2))*np.sin(((3*np.pi)/2)*x)-np.exp(-1/2*np.square(x+2))*np.cos(np.pi*x)





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
    Xp = []
    for i in range(n):
        Xp.append(np.power(x, i))
    return np.array(Xp).T

def polyEqn(x,m):
    y = 0
    for i in range(1,len(m)):
        y = y + m[i]*np.power(x,i)
    return y

def RegressionCoeffiecnts(p):
    X = x_poly(x_training, p)
    m = normal(X, y_training)
    return m

if __name__ == '__main__':

    myplot = plotter()
    x_training,y_training = generateSetRandom(30)

    m = RegressionCoeffiecnts(15)
    x = np.linspace(-5, 5, 1000)
    y_fit = polyEqn(x, m)
    y_real = f(x)
    plt.plot(x,y_real)
    plt.plot(x,y_fit)
    plt.scatter(x_training,y_training)
    plt.ylim(-2, 2)
    plt.xlim(-5, 5)
    plt.show()





