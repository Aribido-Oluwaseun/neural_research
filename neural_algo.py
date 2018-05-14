import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class NeuralAlgo:

    def __init__(self):
        self.i = 3
        self.cols = [0, 2]
        self.data = self.getTrainAndLabel(self.cols)
        np.random.seed(50)

    def getData(self):
        return datasets.load_iris()

    def getTrainAndLabel(self, columns):
        iris = self.getData()
        data = {}
        data['data1'] = np.asarray(iris.data[:, columns[0]])
        data['data2'] = np.asarray(iris.data[:, columns[1]])
        data['label'] = np.asarray(iris.target)
        data = pd.DataFrame.from_dict(data)
        data = data.sample(frac=1).reset_index(drop=True)
        return data

    def plot(self, X, y):
        x_min, x_max = X['data1'].min() - .5, X['data1'].max() + .5
        y_min, y_max = X['data2'].min() - .5, X['data2'].max() + .5
        plt.figure(2, figsize=(8, 6))
        plt.clf()

        # Plot the training points
        plt.scatter(X['data1'], X['data2'], c=y, cmap=plt.cm.Set1)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.title('Scatter Plot of Petal Width vs Sepal Length')
        plt.show()

    def verifyPredictions(self, w_i, data):
        data = data.sample(frac=1).reset_index(drop=True)
        data['predict'] = np.zeros([data.shape[0], 1])
        for i in range(0, data.shape[0]):
            selectedPoints = data.iloc[i, 0:2]
            indices = w_i.index.values.tolist()
            distances = [self.distance(x1=selectedPoints['data1'],
                                       x2=selectedPoints['data2'],
                                       y1=w_i['data1'][x],
                                       y2=w_i['data2'][x]) for x in indices]
            c = distances.index(min(distances))
            data.iloc[i, 3] = c
        diff = (data['label'] == data['predict']) + 0
        count_vals = np.unique(data['predict'], return_counts=True)
        return count_vals, float(np.sum(diff, axis=0)) / data.shape[0]

    def distance(self, x1, x2, y1, y2):
        return np.sqrt(np.square(x1-y1) + np.square(x2-y2))

    def classicKMeans(self, plot=False):
        points = np.random.choice(range(1, self.data.shape[0]), 3)
        w_i = self.data.iloc[points, 0:2]
        w_i.index = range(0, w_i.shape[0])
        n_i = np.zeros([w_i.shape[0], 1])
        colors = ['red', 'blue', 'green']

        # implement the online algorithm
        for i in range(0, self.data.shape[0]):
            selectedPoints = self.data.iloc[i, 0:2]
            indices = w_i.index.values.tolist()
            distances = [self.distance(x1=selectedPoints['data1'],
                                   x2=selectedPoints['data2'],
                                   y1=w_i['data1'][x],
                                   y2=w_i['data2'][x]) for x in indices]
            c = distances.index(min(distances))
            n_i[c] += 1
            w_i.iloc[c, :] = w_i.iloc[c, :] + (1/n_i[c])+(selectedPoints-w_i.iloc[c, :])

            if plot:
                plt.plot([w_i.iloc[c, 0]], [w_i.iloc[c, 1]], marker='x', markersize=3, color=colors[c])
        if plot:
            plt.plot([w_i.iloc[0, 0]], [w_i.iloc[0, 1]], marker='o', markersize=7, color='black')
            plt.plot([w_i.iloc[1, 0]], [w_i.iloc[1, 1]], marker='o', markersize=7, color='black')
            plt.plot([w_i.iloc[2, 0]], [w_i.iloc[2, 1]], marker='o', markersize=7, color='black')
            plt.title('Online K-Means Learning Trajectory')
            X = self.data[['data1', 'data2']]
            y = self.data['label']
            self.plot(self.data[['data1', 'data2']], self.data['label'])
        return w_i

    def neural_algo(self, plot=False):

        # Initialize W, thetha and n
        points = np.random.choice(range(1, self.data.shape[0]), 3)
        W = self.data.iloc[points, 0:2]
        W.index = range(0, W.shape[0])
        thetha = np.zeros([W.shape[0], ])
        n_i = np.ones([W.shape[0], ])  # 1 dim data in numpy appears as np.xxx([dim, ])
        colors = ['red', 'blue', 'green']
        # We iterate over T times
        for T in range(0, self.data.shape[0]):
            x_T = self.data.iloc[T, 0:2]
            c, y_i_t, z_T = self.a_T(W, x_T, thetha)
            n_i[c] = n_i[c] + 1
            W.iloc[c, :] = W.iloc[c, :] + 1 / n_i[c] * (2 * x_T - W.iloc[c, :])
            thetha[c] = thetha[c] + 1 / n_i[c] * (z_T - thetha[c])
            if plot:
                plt.plot([.5 * W.iloc[c, 0]], [.5 * W.iloc[c, 1]], marker='x', markersize=3, color=colors[c])
        if plot:
            plt.plot([.5 * W.iloc[0, 0]], [.5 * W.iloc[0, 1]], marker='o', markersize=7, color='black')
            plt.plot([.5 * W.iloc[1, 0]], [.5 * W.iloc[1, 1]], marker='o', markersize=7, color='black')
            plt.plot([.5 * W.iloc[2, 0]], [.5 * W.iloc[2, 1]], marker='o', markersize=7, color='black')
            plt.title('Neural-algorithm Learning Trajectory')
            X = self.data[['data1', 'data2']]
            y = self.data['label']
            self.kmeans.plot(X, y)
        return 0.5 * W

    def a_T(self, W_T, X_T, thetha_T):
        a_i_T = -np.dot(W_T, X_T) + thetha_T
        c = np.argmin(a_i_T, axis=0)
        y_i_t = np.zeros_like(a_i_T)
        y_i_t[c] = 1
        z_T = -1*a_i_T[c]
        return c, y_i_t, z_T

    def run(self, k, plot=False):
        count_vals_k = list()
        accuracy_k = list()
        count_vals_j = list()
        accuracy_j = list()
        for i in range(0, k):
            W_k = self.neural_algo(plot)
            W_j = self.classicKMeans(plot)
            count_k, acc_k = self.verifyPredictions(W_k, self.data)
            count_j, acc_j = self.verifyPredictions(W_j, self.data)
            count_vals_k.append(count_k[1])
            accuracy_k.append(acc_k)
            indices_k = count_k[0]
            count_vals_j.append(count_j[1])
            accuracy_j.append(acc_j)
            indices_j = count_j[0]
        classes_k = dict()
        classes_j = dict()

        for i in range(3):
            classes_k[i] = [x[i] for x in count_vals_k]
            classes_j[i] = [x[i] for x in count_vals_j]

        print 'classes_k: ', indices_k
        print 'std.dev_k: ', np.std(classes_k[0]), np.std(classes_k[1]), np.std(classes_k[2])
        print 'mean_k: ', np.asarray(sum(count_vals_k)) / k
        print 'accuracy_k: ', sum(accuracy_k)/k

        print ''
        print 'classes_j: ', indices_j
        print 'std.dev_j: ', np.std(classes_j[0]), np.std(classes_j[1]), np.std(classes_j[2])
        print 'mean_j: ', np.asarray(sum(count_vals_j)) / k
        print 'accuracy_j: ', sum(accuracy_j)/k

def main():
    neural = NeuralAlgo()
    neural.run(30, False)

if __name__ == '__main__':
    main()