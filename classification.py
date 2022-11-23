from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d


if __name__ == '__main__':
    data = load_iris()
    print(data.target[[10, 25, 50]])
    print(list(data.target_names))

    #Need to maximise fishers index
    # iris = datasets.load_iris()
    # X = iris.data[:, :2]  # we only take the first two features.
    # y = iris.target