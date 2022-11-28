from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from sklearn.linear_model import LogisticRegression
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

def SKLDA(X,y):
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)

    return X_r2


    lw = 2

def withinClassScatter(matrixClass):
    mean = matrixClass.mean(1).T
    S_k = [0,0,0,0]
    for i in range(len(matrixClass)):
        test = matrixClass[:,i]-mean
        test2 = (matrixClass[:,i] - mean).T


        S_k = S_k +np.outer((matrixClass[:,i]-mean),((matrixClass[:,i] - mean).T))
        print(S_k)
    return S_k

def betweenClassCovariance(class_means):
    class_means = np.array(class_means)
    globalmean = class_means.mean(0)
    S_b = [0,0,0,0]
    for i in range(len(class_means)):
        test = (class_means[i]-globalmean),
        S_b = S_b + np.outer((class_means[i]-globalmean),((class_means[i]-globalmean)).T)
    return S_b


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
        if sample[4] == 1:
            classB = np.r_[classB,sample[:-1]]
        if sample[4] == 2:
             classC = np.r_[classC,sample[:-1]]

    #Tidy up to get rid of this
    classA = classA.reshape(4,50)
    classB = classB.reshape(4, 50)
    classC = classC.reshape(4, 50)

    S_kA = withinClassScatter(classA)

    S_kB = withinClassScatter(classB)

    S_kC = withinClassScatter(classC)

    S_w = S_kA+S_kB+S_kC
    list = (S_kA,S_kB,S_kC)

    mean_A = classA.mean(1)
    mean_B = classB.mean(1)
    mean_C = classC.mean(1)
    class_means = (mean_A, mean_B, mean_C)
    # mean_global = np.average(mean_A,mean_B,)
    S_b = betweenClassCovariance(class_means)
    temp1 = np.linalg.inv(S_w)
    temp = (np.linalg.eig(np.dot(np.linalg.inv(S_w),S_b))[0])
    eigvs = np.linalg.eig(np.dot(np.linalg.inv(S_w),S_b))[1]
    #sorted in function?
    vectors = eigvs[0:2].T

    return np.dot(X,vectors)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(Z):
    test = Z
    denom = 0

    for i in range(len(Z)):
        denom = denom + np.exp(Z[i])

    for j in range(len(Z)):
        test = Z[j]
        for p in range(len(Z[j])):
            test1 = denom[p]
            if denom[p] != 0:
                Z[j][p] = Z[j][p]/denom[p]

    return Z

def encodeClasses(y):
    zero = [0,0,0,1]
    one = [0,0,1,0]
    two = [0,1,0,0]
    three = [1, 0, 0, 0]
    y_encoded = []
    for i in range(len(y)):
        if y[i] == 0:
            y_encoded.append([0,0,0,1])
        elif y[i] == 1:
            y_encoded.append([0,0,1,0])
        elif y[i] == 2:
            y_encoded.append([0,1,0,0])
        elif y[i] == 3:
            y_encoded.append([1, 0, 0, 0])

    return np.stack(y_encoded, axis=0 )

def trainLogistic(X,y,alpha,iterations):
    #Weigths equal number of features
    #Features are x with column for 1s
    y_encoded = encodeClasses(y)
    n_training= len(X)
    features = np.c_[np.ones(len(X)), X]
    weights = np.zeros((features.shape[1],y_encoded.shape[1]))

    for i in range(iterations):
        hypthesis = softmax(-np.dot(features,weights))
        tes1 = X
        test = np.dot(features.T,(y_encoded-hypthesis))
        weights = weights - (alpha/n_training)*np.dot(features.T,(y_encoded-hypthesis))

    return weights

def testLogisticModel(X,y,weights):
    y_encoded = encodeClasses(y)
    features=np.c_[np.ones(len(X)),X]
    linear = np.dot(features,weights)
    probability = softmax(linear)
    prediction = []
    for set in probability:

        prediction.append((np.argmax(set)))
        prediction_encoded = encodeClasses(prediction)
        np.stack(prediction, axis=0)
    results = (y_encoded- prediction_encoded)
    correct  = 0
    for result in results:
        test = result
        if np.all(result==0):
            correct = correct +1
    accuracy = (correct/len(y_test))*100
    return accuracy

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

    X_r2 = SKLDA(X, y)
    X_lda = LDA(X,y)



    f2 = plt.figure(2)
    for color, i, target_names in zip(colors, [0, 1, 2], target_names):
        plt.scatter(
            X_lda[y == i, 0], X_r2[y == i, 1], alpha=0.8, color=color, label=target_names
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("LDA")
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X,y)

    weights = trainLogistic(X_train,y_train,0.01,1000)

    print(testLogisticModel(X_test,y_test,weights))

    # print(model.classes_)
    # print(model.score(X, y))





