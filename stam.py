#@author=Joseph_Aribido
import util
import pandas as pd
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import os
import sys
from scipy.signal import hilbert


SAVE_MODEL_PATH = '/Users/joseph/Desktop/Dev/neural_algo/dataset/pickle_obj'
SAVE_IMAGE_PATH = '/Users/joseph/Desktop/Dev/neural_algo/imgs/'

class STAM_Exception(Exception):
    'Returns and Exception for STAM'


class STAM:

    def __init__(self, num_clusters):
        self.num_clusters = num_clusters

    def make_df(self, data):
        df = pd.DataFrame(data=data, index=range(len(data)))
        return df

    def balanced_order(self, data, lbl):
        data_size = data.shape[0]
        print 'generating balanced order data for experiment...'
        data = pd.DataFrame(data=data, columns=range(data.shape[1]))
        data['lbl'] = lbl
        data_holder = [[] for _ in range(10)]
        for i in range(data.shape[0]):
            data_holder[int(data.iloc[i, -1])].append(data.iloc[i, :])
        df = pd.DataFrame(data=np.zeros([data.shape[0], data.shape[1]]))
        count = 0
        while data_size > 1:
            digits = np.random.choice(a=range(0, 10), size=10, replace=False)
            for j in digits:
                df.iloc[count, :-1] = data_holder[j][np.random.choice(
                    a=range(len(data_holder[j])), size=1, replace=True)[0]]
                df.iloc[count, -1] = int(j)
                count += 1
            data_size -= 10
        data = df
        data = data.sample(frac=1).reset_index(drop=True)
        print 'balanced order data generated!\n'
        return data

    def optimal_initilialization(self, df, num_centroids, m, debug=False):
        W = np.zeros([num_centroids, df.shape[1] - 1])
        classes = range(num_centroids)
        digits = [m for _ in classes]
        while sum(digits) > 0:
            indices = np.random.choice(a=df.index.values, size=num_centroids, replace=False)
            for j in range(W.shape[0]):
                if digits[int(df.iloc[indices[j], -1])] == 0:
                    pass
                else:
                    if debug:
                        print 'index', indices[j]
                        print 'label', df.iloc[indices[j], -1], '\n'
                    W[int(df.iloc[indices[j], -1]), :] = W[int(df.iloc[indices[j], -1]), :] + df.iloc[indices[j], :-1]
                    digits[int(df.iloc[indices[j], -1])] -= 1
        W = W.astype(np.double) / m
        #W = W.astype(np.double)
        W = pd.DataFrame(data=W)
        #W['l'] = range(num_centroids)
        # W = W.sample(frac=1).reset_index(drop=True)
        #label = W['l']
        #W = W.drop('l', 1)
        return W

    def random_order(self, data, lbl):
        print 'please wait, generating random order data...'
        df = pd.DataFrame(data=data, columns=range(data.shape[1]))
        df['lbl'] = lbl
        df = df.sample(frac=1).reset_index(drop=True)
        return df

    def initialize_w(self, W_init, num_centroid, num_feature, mu=0, sigma=1, df=None, m=5):
        l = None
        # We give this option in case we have some learned W for which we want to optimize with further training
        # This is useful if we want to train our algorithm in several epochs.
        if W_init == 'ZERO_INIT':
            print 'initializing W with ZERO digits...'
            W = pd.DataFrame(np.zeros([num_centroid, ]))
        elif W_init == 'RND_INIT':
            print 'initializing W with random digits...'
            W = pd.DataFrame(np.random.rand(num_centroid, num_feature))
        elif W_init == 'RND_INIT_RAW':
            print 'initializing W with random digits...'
            W = pd.DataFrame(np.random.randint(low=0, high=255, size=(num_centroid, num_feature)))
        elif W_init == 'NORM_INIT':
            print 'initializing W with normal RVs...'
            W = pd.DataFrame(data=np.random.normal(mu, sigma, size=(num_centroid, num_feature)))
        elif W_init == 'WHT_INIT':
            W = pd.DataFrame(data=np.random.normal(mu, sigma, size=(num_centroid, num_feature)))
            W = util.Data_Processing().zca_whitening_matrix(W)
            W = pd.DataFrame(data=W)
        elif W_init == 'OPT_INIT':
            # Here, we initialize W with each digit class.
            print 'initializing W with m digit class'
            W = self.optimal_initilialization(df, num_centroids=num_centroid, m=m)
        else:
            raise (STAM_Exception('please specify W_int=RND_INT or W_int=OPT_INT'))
        return W

    def distance(self, img1, img2):
        return np.square(np.linalg.norm((img1-img2), 1))

    def neural_kmeans(self, data, lbl, iter_pred=False, map_matrix=False, order='BALANCED',
                    init_type='RND_INIT', show_image=False, show_centroid=False, m=5):

        matrix = pd.DataFrame(data=np.ones([self.num_clusters, 10]), index=range(self.num_clusters), dtype=int)
        track_pred1 = [np.empty(data.shape[0], ) for _ in range(self.num_clusters)]
        track_pred2 = [np.empty(data.shape[0], ) for _ in range(self.num_clusters)]
        distance1 = np.empty(data.shape[0], )
        distance2 = np.empty(data.shape[0], )
        df = None
        if order == 'BALANCED':
            df = self.balanced_order(data=data, lbl=lbl)
        elif order == 'RANDOM':
            df = self.random_order(data=data, lbl=lbl)
        else:
            df = data
            df['lbl'] = lbl
        print 'proceeding with neural-algorithm...'
        W = self.initialize_w(init_type, num_centroid=self.num_clusters, num_feature=df.shape[1] - 1,
                                 mu=0, sigma=1, df=df, m=m)
        print 'W initialized...\n'
        thetha = np.zeros([W.shape[0], ])
        n_i = np.ones([W.shape[0], ])
        # We iterate over the examples T times. Where T is the length of each example.
        print 'iterating over examples...'

        if show_centroid:
            for i in range(10):
                plt.subplot(2, 5, i + 1)
                plt.imshow(W.iloc[i, :].values.reshape(28, 28), interpolation='nearest')
                plt.title(str(i))
                plt.axis('off')
            plt.show()

        for T in range(df.shape[0]):
            x_T = df.iloc[T, :-1]
            if show_image:
                if T < 10:
                    plt.subplot(2, 5, T + 1)
                    if show_image:
                        plt.imshow(x_T.values.reshape(28, 28), interpolation='nearest')
                        plt.title(int(df.iloc[T, -1]))
                    plt.axis('off')
            if T == 10:
                plt.show()
            c, y_i_t, z_T = self.a_T(W, x_T, thetha)
            n_i[c] = n_i[c] + 1
            W.iloc[c, :] = W.iloc[c, :] + 1 / n_i[c] * (2 * x_T - W.iloc[c, :])
            thetha[c] = thetha[c] + 1 / n_i[c] * (z_T - thetha[c])
            distance1[T] = min([self.distance(x_T, W.iloc[j, :]) for j in range(self.num_clusters)])
            distance2[T] = self.distance(x_T, W.iloc[c, :])
            if map_matrix:
                matrix.iloc[c, int(df.iloc[T, -1])] += 1
            if iter_pred:
                for i in range(self.num_clusters):
                    track_pred1[i][T] = float(matrix.iloc[i, i])/sum(matrix.iloc[i, :])
                    track_pred2[i][T] = float(max(matrix.iloc[i, :]))/sum(matrix.iloc[i, :])
        print 'neural algorithm completed! \n'
        return 0.5 * W, matrix, track_pred1, track_pred2, distance1, distance2

    def a_T(self, W_T, X_T, thetha_T):
        a_i_T = -np.dot(W_T, X_T) + thetha_T
        c = np.argmin(a_i_T, axis=0)
        y_i_t = np.zeros_like(a_i_T)
        y_i_t[c] = 1
        z_T = -1*a_i_T[c]
        return c, y_i_t, z_T


class Experiments:
    """ This class takes in all experiment attributes needed to run an experiment on a 1-layer stam. """

    def __init__(self, num_train, num_test, num_valid=10000, num_clusters=10, order='RANDOM', init_type='RND_INIT',
                 data_type='minst', processing='scaled', iter_pred=True, map_matrix=True, save_matrix_loc=None,
                 save_centroid_loc=None, show_image=False, show_centroid=False, m=5):
        self.num_train = num_train
        self.num_test = num_test
        self.num_valid = num_valid
        self.order = order
        self.data_type = data_type
        self.processing = processing
        self.pred = iter_pred
        self.init_type = init_type
        self.map_matrix = map_matrix
        self.num_clusters = num_clusters
        self.save_matrix = save_matrix_loc
        self.save_centroid_loc = save_centroid_loc
        self.show_image = show_image
        self.show_centroid = show_centroid
        self.initial_labels = m
        data_proc = util.Data_Processing(num_train=self.num_train, num_test=self.num_test, num_valid=self.num_valid,
                                         data_type=self.data_type, preprocessing=self.processing)
        self.train_img, self.train_lbl, _, _ = data_proc.run()

    def run(self):
        stam = STAM(num_clusters=self.num_clusters)
        W, matrix, track_pred1, track_pred2, distance1, distance2 = stam.neural_kmeans(self.train_img, self.train_lbl, iter_pred=self.pred,
                                                   map_matrix=self.map_matrix, order=self.order,
                                                   init_type=self.init_type, show_image=self.show_image,
                                                   show_centroid=self.show_centroid, m=self.initial_labels)
        if self.save_matrix is not None:
            matrix = matrix.subtract(pd.DataFrame(np.ones_like(matrix.values)))
            print matrix
            writer = pd.ExcelWriter(self.save_matrix)
            matrix.to_excel(writer, 'Sheet1')
            writer.save()

        plt.figure(0)
        if self.save_centroid_loc is not None:
            for i in range(W.values.shape[0]):
                plt.subplot(2, 5, i+1)
                plt.imshow(W.iloc[i, :].values.reshape(28, 28), interpolation='nearest')
                plt.axis('off')
            plt.savefig(self.save_centroid_loc)

        if self.pred:
            plt.figure(1)
            for i in range(self.num_clusters):
                plt.plot(track_pred1[i])
            plt.title('Cluster Accuracy based on Initialization of the clusters')
            plt.legend(['cluster: {}'.format(i) for i in range(self.num_clusters)])
            plt.savefig(SAVE_IMAGE_PATH + 'cl_acc_int')
            plt.show()

            plt.figure(2)
            for i in range(self.num_clusters):
                plt.plot(track_pred2[i])
            plt.title('Cluster Accuracy based on the Best clusters')
            plt.legend(['cluster: {}'.format(i) for i in range(self.num_clusters)])
            plt.savefig(SAVE_IMAGE_PATH + 'cl_acc_best_int')
            plt.show()

            plt.figure(3)
            distance1 = self.moving_average(distance1, 10)
            distance2 = self.moving_average(distance2, 10)
            plt.plot(list(distance1)[8000:8500])
            plt.plot(list(distance2)[8000:8500])
            plt.legend(['2 norm distance', 'selected centroid distance'])
            plt.savefig(SAVE_IMAGE_PATH + 'dist_norm_vs_algo')
            plt.show()

    def moving_average(self, values, window):
        weights = np.repeat(1.0, window) / window
        sma = np.convolve(values, weights, 'valid')
        return sma


def main():
    num_train = 10000
    num_test = 10
    num_valid = 10
    num_clusters = 10
    order = 'RANDOM'
    init_type = 'OPT_INIT'  # our prefered intialialization
    data_type = 'mnist'
    processing = 'raw'
    iter_pred = True
    map_matrix = True
    save_matrix_loc = SAVE_IMAGE_PATH + order + '_' + init_type + '_.xlsx'
    save_centroid_loc = SAVE_IMAGE_PATH + order + '_' + init_type + '_.png'
    show_image = False
    show_centroid = False
    init_labels = 5
    expt = Experiments(num_train, num_test, num_valid=num_valid, num_clusters=num_clusters, order=order,
                       init_type=init_type, data_type=data_type, processing=processing, iter_pred=iter_pred,
                       map_matrix=map_matrix, save_matrix_loc=save_matrix_loc, save_centroid_loc=save_centroid_loc,
                       show_image=show_image, show_centroid=show_centroid, m=init_labels)
    expt.run()

if __name__=='__main__':
    main()