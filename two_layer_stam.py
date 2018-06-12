#@author=Joseph_Aribido

import numpy as np
import pandas as pd
import util
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap as ismp

class Hierarchy_STAM_Exception(Exception):
    "Defines errors related to Hierarchical_STAMs"

class Extract_Data():

    def __init__(self, num_train=10000, num_test=1000, num_valid=1000):
        self.num_train = num_train
        self.num_test = num_test
        self.num_valid = num_valid
        train_img, train_lbl, test_img, test_lbl = self.get_data()

        self.valid_img = train_img[self.num_train: self.num_train + self.num_valid]
        self.valid_lbl = train_lbl[self.num_train: self.num_train + self.num_valid]

        self.train_img = train_img[0: self.num_train]
        self.train_lbl = train_lbl[0: self.num_train]

        self.test_img = test_img[0: self.num_test]
        self.test_lbl = test_lbl[0: self.num_test]

        self.tract_predictions = list()
        self.mse_accuracy = list()
        self.simm_accuracy = list()
        self.cluster_labels = ['cluster A', 'cluster B', 'cluster C', 'cluster D', 'cluster E', 'cluster F',
                               'cluster G', 'cluster H', 'cluster k', 'cluster L']

    def get_data(self):
        return util.load()

    def get_cluster_labels(self, num):
        return self.cluster_labels[0: num]

    def get_prep_data(self, type='train'):
        if type == 'train':
            return self.make_df(self.train_img), self.train_lbl
        elif type == 'test':
            return self.make_df(self.test_img), self.test_lbl
        elif type == 'validation':
            return self.make_df(self.valid_img), self.valid_lbl
        else:
            return 0

    def get_train_data(self):
        return self.train_lbl

    def test(self):
        print self.test_lbl

    def make_df(self, data):
        df = pd.DataFrame(data=data, index=range(len(data)))
        return df


class Create_RFs:
    """This class takes one image and creates m receptive fields.
    Take the size of the receptive field to be x,
    """
    def __init__(self, image, image_id, rf_size=4, len_stride=2, num_clusters=3):
        self.image = np.asarray(image).astype(np.double).reshape(28, 28)
        self.rf_size = rf_size
        self.len_stride = len_stride
        self.num_clusters = num_clusters
        self.image_id = image_id

    def run(self):
        return self.create_receptive_field(image=self.image, image_id=self.image_id, rf_size=self.rf_size,
                                           stride=self.len_stride)

    def create_receptive_field(self, image, image_id, rf_size, stride, col=28, row=28):
        store_rf = list()
        image = np.asarray(image).reshape(col, row)
        track_row = 0
        image = pd.DataFrame(data=image)
        while (track_row < image.shape[0] - rf_size + 1):
            track_col = 0
            while (track_col < image.shape[0] - rf_size + 1):
                rf = image.iloc[track_row:track_row + rf_size, track_col:track_col + rf_size]
                track_col += stride
                store_rf.append(list(rf.values.reshape(1, rf_size * rf_size)))
            track_row += stride
        return store_rf, image_id


class Stam:

    def __init__(self, data, num_clusters, stam_id, W=None, lbl=None, mapping_matrix=False, accuracy=0):
        """Please note here that the values and objects passed into this constructor
         class must be processed  before it is passsed in. We try to avoid any data p
         rocessing in the Stam class for performance purpose"""
        self.data = data
        self.num_clusters = num_clusters
        self.stam_id = stam_id
        self.W = W
        self.lbl = lbl
        self.mapping_matrix = mapping_matrix
        self.accuracy = accuracy

    def run(self, data=None, num_clusters=None, W=None):
        if data is None:
            data = self.data
        if num_clusters is None:
            num_clusters = self.num_clusters
        if W is None:
            W = self.W
        return self.stam(df=data,
                         num_clusters=num_clusters,
                         stam_id=self.stam_id, W=W,
                         lbl=self.lbl, mapping_matrix=self.mapping_matrix)

    def stam(self, df, num_clusters, stam_id, W, lbl=None, mapping_matrix=False):
        map_int = range(4)
        matrix = pd.DataFrame(data=np.zeros([num_clusters, len(map_int)]), dtype=int)
        if W is None:
            W = pd.DataFrame(np.random.rand(num_clusters, df.shape[1]))
        thetha = np.zeros([W.shape[0], ])
        n_i = np.ones([W.shape[0], ])

        for T in df.index.tolist():
            x_T = df.iloc[T, :]
            c, y_i_t, z_T = self.a_T(W, x_T, thetha)
            W.iloc[c, :] = W.iloc[c, :] + 1 / n_i[c] * (2 * x_T - W.iloc[c, :])
            thetha[c] = thetha[c] + 1 / n_i[c] * (z_T - thetha[c])
            n_i[c] = n_i[c] + 1
            if mapping_matrix:
                    matrix.iloc[c, lbl[0]] += 1
        print 'stams {} completed!\n'.format(self.stam_id)

        # Here we attempt to get the nth best cluster using the mapping matrix
        if mapping_matrix:
            best_cluster = matrix[lbl[0]].nlargest(n=self.num_clusters, keep='first')
            best_cluster = best_cluster.index.tolist()[self.accuracy]
            return 0.5 * W, matrix, W.loc[best_cluster]
        return 0.5 * W

    def a_T(self, W_T, X_T, thetha_T):
        a_i_T = -np.dot(W_T, X_T) + thetha_T
        c = np.argmin(a_i_T, axis=0)
        y_i_t = np.zeros_like(a_i_T)
        y_i_t[c] = 1
        z_T = -1*a_i_T[c]
        return c, y_i_t, z_T


class Hierarchy_STAMS:

    def __init__(self, data, lbl,  stride, num_clusters, layers=2, neurons=(5, 10), red_dim=(0, 4), rf_size=(4,4)):
        self.data = data
        self.lbl = lbl
        self.rf_size = rf_size
        self.stride = stride
        self.num_clusters = num_clusters
        self.layers = layers
        self.neurons = neurons
        self.num_neighbors = 4
        self.red_dim = red_dim

        if not (isinstance(neurons, list) and len(neurons) == layers):
            raise(Hierarchy_STAM_Exception('configuration passed is incorrect!'
                                              'neurons must be a list() and layers must equal the number of'
                                              'neurons...'))

    def arrange_image_parts_into_RFs(self, data, rf_size, num_clusters):
        # note that we will create a number of receptive field that would each act on the same part of the data.
        # create DataFrames equal to the number of RFs
        data_rf = Create_RFs(image=data.iloc[0, :], image_id=None, rf_size=rf_size,
                             len_stride=self.stride, num_clusters=num_clusters).run()
        partition_rf = [list() for _ in range(len(data_rf))]
        # Arrange parts of the images into several RFs to be fed into the STAM modules
        for i in data.index.tolist():
            data_rf = Create_RFs(image=data.iloc[i, :], image_id=None, rf_size=rf_size,
                                 len_stride=self.stride, num_clusters=self.num_clusters).run()
            for k in range(len(data_rf)):
                partition_rf[k].append(data_rf[k][0])
        partition_rf = [pd.DataFrame(
            data=np.asarray(partition_rf[k]).reshape(data.shape[0], len(data_rf[0][0])))
            for k in range(len(partition_rf))]
        return partition_rf

    def reduce_dimension(self, data, num_neighbors=10, dim=4, type='ave'):
        """This function reduces the dimension of data passed into it. Data is a list containing
        numpy array data types"""
        if type == 'ave':
            data = data.mean(axis=1)
        elif type == 'isomap':
            imap = ismp(n_neighbors=num_neighbors, n_components=dim, n_jobs=-1)
            imap = imap.fit(data)
            data = imap.fit_transform(data)
        return data

    def run(self, data, lbl, layer, neurons, rf_size, red_dim, mapping_matrix=False, accuracy=0):
        """This method implements hierarchical STAMs.
        layers: indicates how many STAM layers we need
        num-of_neurons: is a list object that takes in the number of neurons at each layer. e.g [2, 10] implies 2 neuron
        output per STAM module at the first layer and 10 neurons at the second layer.
        """
        print 'running stam heirarchy algorithm...'
        while layer > 1:
            # # create stam receptive fields
            store_W = dict()
            store_matrix = dict()
            store_best_cluster = dict()

            # print some stats
            row = int(np.sqrt(len(data.iloc[0, :])))
            print 'stats...'
            print 'size per image = {} by {}'.format(row, row)
            print 'receptive field size = {}'.format(rf_size)
            print 'arranging parts of the images into several RFs to be fed into STAM modules...'
            partition_rf = self.arrange_image_parts_into_RFs(data=data)
            print 'done!\n'
            #create modules for the various RFs
            print 'creating stam modules for respective receptive fields...'
            stams = [Stam(data=partition_rf[k], lbl=lbl, num_clusters=neurons, stam_id=k, W=None,
                          mapping_matrix=mapping_matrix, accuracy=accuracy) for k in range(len(partition_rf))]
            print 'running stam modules for respective receptive fields...'
            if mapping_matrix:
                for i in range(len(stams)):
                    store_W[i], _, store_best_cluster[i] = stams[i].run()
            else:
                for i in range(len(stams)):
                    store_W[i] = stams[i].run()
            # Reduce the dimension of the best clusters
            data = np.asarray([np.asarray(store_best_cluster[i]) for i in range(len(store_best_cluster))])
            data = data.reshape(len(data), len(data[0]))
            data = self.reduce_dimension(data=data, num_neighbors=self.num_neighbors, dim=red_dim)
            layer -= 1
            return self.run(data=data, lbl=None, layer=self.layers[layer], neurons=self.neurons[layer],
                            rf_size=self.rf_size[layer], red_dim=self.red_dim[layer], mapping_matrix=False, accuracy=0)
        else:
            store_W, store_matrix, store_best_cluster = self.run(data=data, lbl=None, layer=self.layers[layer], neurons=self.neurons[layer],
                     rf_size=self.rf_size[layer], red_dim=self.red_dim[layer], mapping_matrix=False, accuracy=0)


    def experiments(self, mapping_matrix=False, digit=0, accuracy=0):
        data, lbl = self.test_1_digit(digit=digit)
        return self.run(data=data, lbl=lbl, mapping_matrix=mapping_matrix, accuracy=accuracy)


    def test_1_digit(self, digit):
        print 'generating 1 digit images for experiment...'
        data_holder = list()
        lbl_holder = list()
        for i in range(self.data.shape[0]):
            if self.lbl[i] == digit:
                data_holder.append(self.data.iloc[i, :])
                lbl_holder.append(self.lbl[i])
        data_holder = np.asarray(data_holder).reshape(len(data_holder), len(data_holder[0]))
        data_holder = pd.DataFrame(data=data_holder)
        print 'images generated!\n'
        return data_holder, lbl_holder

def run_experiments():
    data = Extract_Data()
    data, lbl = data.get_prep_data(type='train')
    hierarchy_stams = Hierarchy_STAMS(data=data, lbl=lbl, rf_size=4, stride=4, num_clusters=10)


def main():
    pass

if __name__ == '__main__':
    main()