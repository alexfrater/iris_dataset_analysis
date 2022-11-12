
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

def f(x):
    return np.exp(-1/2*np.square(x-2))*np.sin(((3*np.pi)/2)*x)-np.exp(-1/2*np.square(x+2))*np.cos(np.pi*x)

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
def polyEqn(x,p):
    y = 0
    for i in range(p):
        y = y + np.power(x,i+1)
    return  y

if __name__ == '__main__':


    x_training,y_training = generateSetArranged(30)
    #scatterPlot(x_training,y_training)
    X = x_poly(x_training,5)

    out = normal(X,y_training)



    print(out)





