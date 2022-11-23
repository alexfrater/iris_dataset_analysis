from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def SKLDA(X,y):
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)

    return X_r2
    lw = 2

def LDA(X,y):
    # y = y.t
    # data = np.hstack([X, y])
    data = np.c_[X,y]

    classA = []
    classB = []
    classC = []
    for i in range(150):
        sample = data[i]
        if sample[4] == 0:
            classA = np.r_[classA,sample[:-1]]
        # if sample[4] == 1:
        #     classB = np.r_[classB,sample[:-1]]
        # if sample[4] == 2:
        #     classC = np.r_[classC,sample[:-1]]

    #Tidy up to get rid of this
    classA = classA.reshape(4,50)
    classB = classA.reshape(4, 50)
    classC = classA.reshape(4, 50)

    numA= classA.shape[0]
    S_kA = np.var(classA) * numA

    numB = classB.shape[0]
    S_kB = np.var(classB) * numB

    numC = classC.shape[0]
    S_kC = np.var(classC) * numC

    S_w = S_kA+S_kB+S_kC
    list = (S_kA,S_kB,S_kC)

    mean_A = np.average(classA)
    mean_B = np.average(classB)
    mean_C = np.average(classC)
    class_means = (mean_A, mean_B, mean_C)
    # mean_global = np.average(mean_A,mean_B,)
    S_B = np.var(class_means)

    proj = np.maximum(np.linalg.eig(np.dot(np.linalg.inv(S_w),S_B)))





if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    colors = ["blue", "green", "red"]
    f1 = plt.figure(1)
    for color, i, target_names in zip(colors, [0, 1, 2], target_names):
        plt.scatter(
            X[y == i, 0], X[y == i, 1], alpha=0.8, color=color, label=target_names
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("Original Data")

    X_lda = LDA(X,y)

    X_r2 = SKLDA(X,y)

    f2 = plt.figure(2)
    for color, i, target_names in zip(colors, [0, 1, 2], target_names):
        plt.scatter(
            X_r2[y == i, 0], X_r2[y == i, 1], alpha=0.8, color=color, label=target_names
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("LDA")
    plt.show()





