#@author=Joseph_Aribido

import numpy as np
import pandas as pd
import util
import matplotlib.pyplot as plt
from warnings import warn
from sklearn.manifold import Isomap as ismp

class Hierarchy_STAM_Exception(Exception):
    "Defines errors related to Hierarchical_STAMs"

class Extract_Data():

    def __init__(self, num_train=10, num_test=1000, num_valid=1000):
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

class Stam:

    def __init__(self, num_clusters, stam_id, rf_size, mapping_matrix=True):
        self.num_clusters = num_clusters
        self.thetha = np.zeros([self.num_clusters, ])
        self.n_i = np.ones([self.num_clusters, ])
        self.best_cluster = None
        self.stam_id = stam_id
        self.rf_size = rf_size
        self.matrix = pd.DataFrame(data=np.zeros([self.num_clusters, 10]))
        self.W = pd.DataFrame(data=np.zeros([self.num_clusters, self.rf_size*self.rf_size]))
        self.mapping_matrix = mapping_matrix

    def __call__(self, image, lbl, accuracy=0):
        self.stam(image, lbl, accuracy)
        return self.best_cluster

    def stam(self, image, lbl, accuracy):
        image = np.asarray(image)  # convert image into np array if it is not an np array yet
        image = image.reshape(image.shape[1],)
        x_T = np.asarray(image)
        c, y_i_t, z_T = self.a_T(self.W, x_T, self.thetha)
        self.W.iloc[c, :] = self.W.iloc[c, :] + 1 / self.n_i[c] * (2 * x_T - self.W.iloc[c, :])
        self.thetha[c] = self.thetha[c] + 1 / self.n_i[c] * (z_T - self.thetha[c])
        self.n_i[c] = self.n_i[c] + 1
        if self.mapping_matrix:
            self.matrix.iloc[c, lbl] += 1

        # Here we attempt to get the nth best cluster using the mapping matrix
        if self.mapping_matrix:
            self.best_cluster = self.matrix[lbl].nlargest(n=self.num_clusters, keep='first')
            self.best_cluster = self.best_cluster.index.tolist()[accuracy]
            self.best_cluster = self.W.loc[self.best_cluster]
            return self.best_cluster
        else:
            self.get_W()

    def a_T(self, W_T, X_T, thetha_T):
        a_i_T = -np.dot(W_T, X_T) + thetha_T
        c = np.argmin(a_i_T, axis=0)
        y_i_t = np.zeros_like(a_i_T)
        y_i_t[c] = 1
        z_T = -1 * a_i_T[c]
        return c, y_i_t, z_T

    def get_best_cluster(self):
        return self.best_cluster

    def get_all(self):
        return 0.5 * self.W, self.matrix, self.W.loc[self.best_cluster]

    def get_W(self):
        return 0.5 * self.W

    def get_matrix(self):
        return self.matrix

class Hierarchy_STAMS:

    def __init__(self, data, lbl,  stride=(2, 1), layers=2, neurons=(5, 10), dim=(0, 4), rf_size=(4, 18),
                 red_dim=(True, False), mapping_matrix=(True, False)):
        self.data = data
        self.lbl = lbl
        self.rf_size = rf_size
        self.stride = stride
        self.layers = layers
        self.neurons = neurons
        self.num_neighbors = 4
        self.dim = dim
        self.mapping_matrix = mapping_matrix
        self.red_dim = red_dim
        self.weights = {}
        self.matrices = {}
        self.stams = {}
        self.n_i = {}
        self.layer = 0
        self.intilialize_paramters()

    def intilialize_paramters(self):
        n = (int(np.sqrt(len(self.data.iloc[0, :]))), 18)
        q = {}
        rf = {}
        for i in range(self.layers):
            q[i] = int((n[i] - self.rf_size[i]) / self.stride[i] + 1) ** 2
            rf[i] = self.create_receptive_field(image=np.random.rand(1, int(np.square(n[i]))), rf_size=self.rf_size[i], stride=self.stride[i])
            if q[i] != len(rf[i]):
                raise (ValueError('consider changing the rf size.'
                                  'the present rf size will not form a square image!'))
        for i in range(self.layers):
           self.stams[i] = [Stam(num_clusters=self.neurons[i], stam_id=k, rf_size=self.rf_size[i],
                                      mapping_matrix=self.mapping_matrix[i]) for k in range(q[i])]

    def learn(self, image, lbl, layer, dim=2, red_dim=True):
        images = self.create_receptive_field(image=image, rf_size=self.rf_size[layer], stride=self.stride[layer])
        self.weights[layer] = [self.stams[layer][k](images[k], lbl) for k in range(len(self.stams[layer]))]
        if red_dim or layer == self.layers-1:  # if are at the last layer, don't reduce dimension.
            self.weights[layer] = [self.reduce_dimension(self.weights[layer][k], dim[layer]) for k in range(len(self.weights[layer]))]
        assembled_image_vector = np.asarray(self.weights[layer]).reshape(1, len(self.weights[layer][0])*len(self.weights[layer]))
        return assembled_image_vector

    def run(self, data, lbl):
        layer = 0
        for t in range(data.shape[0]):
            if t%500 == 0:
                print '{} images completed!'.format(t)
            image = [data.iloc[t, :]]
            while layer < self.layers:
                image = self.learn(image=image, lbl=lbl[t], layer=layer, dim=self.dim[layer], red_dim=self.red_dim[layer])
                layer += 1

    def get_weights(self, layer):
        return self.weights[layer]

    def create_receptive_field(self, image, rf_size, stride):
        store_rf = list()
        image= np.asarray(image)
        row = int(np.sqrt(image.shape[1]))
        image = image.reshape(row, row)
        track_row = 0
        image = pd.DataFrame(data=image)
        if image.shape[0] == rf_size:
             return [image.values.reshape(1, row*row)]
        while (track_row < image.shape[0] - rf_size + 1):
            track_col = 0
            while (track_col < image.shape[0] - rf_size + 1):
                rf = image.iloc[track_row:track_row + rf_size, track_col:track_col + rf_size]
                track_col += stride
                store_rf.append(list(rf.values.reshape(1, rf_size * rf_size)))
            track_row += stride
        return store_rf

    def reduce_dimension(self, image, dim=2):
        """This function reduces the dimension of data passed into it. Data is a list containing
        numpy array data types"""
        # create an empty pd of the desired DataFrame
        actual_dim = int(np.sqrt(len(image)))  # note that each row of the dataframe is an image.
        desired_dim = actual_dim / dim
        if actual_dim % dim != 0:
            raise (ValueError('incompatible dimension specified!'))
        if dim < 2:
            return np.mean(image)
        temp = np.zeros([dim, dim])
        new_data = np.array(image).reshape(actual_dim, actual_dim)
        for j in range(dim):
            for k in range(dim):
                temp[j, k] = np.mean(new_data[j * desired_dim:j * desired_dim + desired_dim,
                                         k * desired_dim:k * desired_dim + desired_dim])
        return temp.reshape(1, dim*dim)[0]


def main():
    data = Extract_Data()
    data, lbl = data.get_prep_data(type='train')
    hierarchy_stams = Hierarchy_STAMS(data, lbl,  stride=(2, 1), layers=2, neurons=(5, 10), dim=(0, 4), rf_size=(4, 18),
                 red_dim=(True, False), mapping_matrix=(True, False))
    hierarchy_stams.run(data, lbl)
    hierarchy_stams.get_weights(1)

if __name__ == '__main__':
    main()