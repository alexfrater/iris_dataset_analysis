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
    classmean = np.mean(matrixClass.T, axis = 0)

    S_k2 = np.dot((matrixClass.T - classmean).T,(matrixClass.T-classmean))
    return S_k2

def betweenClassCovariance(classA,classB,classC):
    mean_A = classA.mean(1)
    mean_B = classB.mean(1)
    mean_C = classC.mean(1)

    class_means = np.array([mean_A, mean_B, mean_C])

    globalmean = np.mean(class_means, axis = 0)

    S_b2 = 50 * np.dot((class_means - globalmean).T,(class_means-globalmean))
    return S_b2


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

    #FIX
    S_b = betweenClassCovariance(classA,classB,classC)
    temp1 = np.linalg.inv(S_w)
    temp = (np.linalg.eig(np.dot(np.linalg.inv(S_w),S_b)))
    eigvs = np.linalg.eig(np.dot(np.linalg.inv(S_w),S_b))[1]
    #sorted in function?
    vectors = eigvs[0:2].T

    return np.dot(X,vectors)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(Z):
    denom = 0
    output_array = []
    for i in range(len(Z)):

        test = Z[i]
        for value in Z[i]:
            denom = denom + np.exp(value)
        newArray = []
        for value in Z[i]:
            newArray.append(np.exp(value)/denom)
        output_array.append(newArray)
        denom = 0

    return output_array

def encodeClasses(y):

    y_encoded = []
    for i in range(len(y)):
        if y[i] == 0:
            y_encoded.append([1,0,0,0])
        elif y[i] == 1:
            y_encoded.append([0,1,0,0])
        elif y[i] == 2:
            y_encoded.append([0,0,1,0])
        elif y[i] == 3:
            y_encoded.append([0,0,0,1])

    return np.stack(y_encoded, axis=0 )


def returnargmax(val):
    out =[]
    for v in val:
        out.append(np.argmax(v))
    return np.array(out)

def trainLogistic(X,y,alpha,iterations):
    #Weigths equal number of features
    #Features are x with column for 1s
    y_encoded = encodeClasses(y)
    n_training= len(X)
    features = np.c_[np.ones(len(X)), X]
    weights = np.zeros((features.shape[1],y_encoded.shape[1]))

    for i in range(iterations):
        linear = -np.dot(features,weights)
        hypthesis = softmax(linear)

        predicted = returnargmax(hypthesis)
        y_actual = y
        out = predicted - y_actual


        weights = weights - (alpha/n_training)*np.dot(features.T,(y_encoded-hypthesis))

    return weights

def testLogisticModel(X,y,weights):
    y_encoded = encodeClasses(y)
    features=np.c_[np.ones(len(X)),X]
    linear = -np.dot(features,weights)
    probability = softmax(linear)
    prediction = []
    for set in probability:
        prediction.append(np.argmax(set))
        # test = np.argmax(set)
        # out = y-test
        # prediction.append((np.argmax(set)))
        # prediction_encoded = encodeClasses(prediction)
        # np.stack(prediction, axis=0)
    test1 = np.array(prediction)
    test2 = y
    out = prediction - y

    # results = (y_encoded- prediction_encoded)
    # correct  = 0
    #
    #
    # for result in results:
    #     test = result
    #     if np.all(result==0):
    #         correct = correct +1
    accuracy = (correct/len(y_test))*100
    return accuracy


def plotData(data,name,target_names,colors,pltnumber):
    plt.figure(pltnumber)
    for color, i, target_names in zip(colors, [0, 1, 2], target_names):
        plt.scatter(
            data[y == i, 0], data[y == i, 1], alpha=0.8, color=color, label=target_names
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title(name)


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    colors = ["blue", "green", "purple"]
    X_r2 = SKLDA(X, y)
    X_lda = LDA(X,y)

    # plotData(X,"Data", target_names,colors,1)
    # plotData(X_r2, "LDASK", target_names, colors,2)
    # plotData(X_lda, "LDA", target_names, colors,3)
    # plt.show()


    X_train, X_test, y_train, y_test = train_test_split(X_r2,y)

    weights = trainLogistic(X_train,y_train,0.01,1000)

    print(str(testLogisticModel(X_test,y_test,weights)), "%")

    # print(model.classes_)
    # print(model.score(X, y))





