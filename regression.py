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

def normal(X,Y,R):

    size = np.shape(X.T)
    test = R*np.identity(1) + np.dot(X.T,X)
    test1 = np.dot(X.T,X)
    return np.dot(np.dot(X.T,Y), la.inv(test))

def x_poly(x,n):
    Xp = []
    for i in range(0,n+1):
        Xp.append(np.power(x, i))
    return np.array(Xp).T

def polyEqn(x,m):
    y = 0
    for i in range(0,len(m)):
        y = y + m[i]*np.power(x,i)
    return y

def RegressionCoeffiecnts(p,R = None):
    if R is None:
        R = 0
    X = x_poly(x_training, p)
    m = normal(X, y_training,R)
    return m

def mse(y_hat, y):
    return np.average(np.square(np.linalg.norm(np.square(y_hat - y))))

if __name__ == '__main__':

    x_training,y_training = generateSetRandom(30)
    reg = 2
    error = 10000000000

    x = np.linspace(-5, 5, 1000)

    y_fit = 0
    y_real = f(x)
    polynomial = []
    polynomial_error = []
    for i in range(0,30):
        sample_m = RegressionCoeffiecnts(i,reg)
        sample_y_fit = polyEqn(x, sample_m)
        sample_error = mse(sample_y_fit - y_real)

        plt.scatter(x_training, y_training)
        polynomial_error.append(sample_error)
        polynomial.append(i)
        plt.plot(x, sample_y_fit)
        plt.ylim(-2, 2)
        plt.xlim(-5, 5)
        plt.show()
        if sample_error < error:
            error = sample_error
            y_fit = sample_y_fit
            m = sample_m


    plt.plot(x,y_real)
    plt.plot(x,y_fit)


    plt.scatter(x_training,y_training)
    plt.ylim(-2, 2)
    plt.xlim(-5, 5)
    plt.show()

    plt.plot(polynomial,polynomial_error)
    plt.show()




