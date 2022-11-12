import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la


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

    x_training,y_training = generateSetRandom(30)

    m = RegressionCoeffiecnts(16)
    x = np.linspace(-5, 5, 1000)
    y_fit = polyEqn(x, m)
    y_real = f(x)
    plt.plot(x,y_real)
    plt.plot(x,y_fit)

    error = np.linalg.norm(y_fit-y_real)
    print(error)
    plt.scatter(x_training,y_training)
    plt.ylim(-2, 2)
    plt.xlim(-5, 5)
    plt.show()





