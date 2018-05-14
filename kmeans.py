
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from matplotlib.path import Path

class Kmeans:

    def __init__(self):
        self.i = 3
        self.cols = [0, 2]
        self.data = self.getTrainAndLabel(self.cols)
        np.random.seed(50)
        # initialize w_i, n_i


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

        return count_vals, float(np.sum(diff, axis=0))/data.shape[0]

    def distance(self, x1, x2, y1, y2):
        return np.sqrt(np.square(x1-y1) + np.square(x2-y2))

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