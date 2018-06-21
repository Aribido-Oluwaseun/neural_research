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

    def __init__(self, num_train=1000, num_test=1000, num_valid=1000):
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

    def __init__(self, num_clusters, stam_id, rf_size, mapping_matrix=True, W_init='random'):
        self.num_clusters = num_clusters
        self.thetha = np.zeros([self.num_clusters, ])
        self.n_i = np.ones([self.num_clusters, ])
        self.best_cluster = None
        self.stam_id = stam_id
        self.rf_size = rf_size
        self.matrix = pd.DataFrame(data=np.zeros([self.num_clusters, 10]))
        if W_init == 'zero':
            self.W = pd.DataFrame(data=np.zeros([self.num_clusters, self.rf_size*self.rf_size]))
        elif W_init == 'random':
            self.W = pd.DataFrame(data=np.random.rand(self.num_clusters, self.rf_size * self.rf_size))
        self.mapping_matrix = mapping_matrix

    def __call__(self, image, lbl, accuracy=0):
        stam_output = self.stam(image, lbl, accuracy)
        return stam_output

    def stam(self, image, lbl, accuracy):
        # convert image into np array if it is not an np array yet
        if len(image.shape) > 1:
            image = image.reshape(max(image.shape),)
        x_T = image
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
            return 0.5 * self.W

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

    def __init__(self, data, lbl,  stride=(2, 1), layers=2, neurons=(5, 10), dim=(2, 1), rf_size=(4, 26),
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
        self.output_centroids = None

    def intilialize_paramters(self):
        n_li = int(np.sqrt(len(self.data.iloc[0, :])))  # 28 by 28 size images in layer 1
        x_li = self.rf_size[0]
        s_li = self.stride[0]
        d_mi = self.dim[0]
        q = {}
        rf = {}
        rf[0] = self.rf_size[0]
        i = 0
        while i < self.layers:
            if (self.red_dim[i] is not False) and (self.dim is not None):
                q[i] = int(((n_li-self.rf_size[i])/self.stride[i])+1)**2
                rf[i+1] = int(np.sqrt(((self.rf_size[i]**2)*q[i])/(self.dim[i])))
                self.rf_size[i+1] = rf[i+1]
                # now compare calculated values with inputed value. They should match.
                if rf[i+1] != self.rf_size[i+1]:
                    raise(Hierarchy_STAM_Exception('Please check the calculation of RF size for layer {}'.format(i+1)))
            else:
                q[i] = 1
                rf[i] = 1
                self.stride[i] = 1
            i += 1

        ## Make this automatic later.

        for i in range(self.layers):
           self.stams[i] = [Stam(num_clusters=self.neurons[i], stam_id=k, rf_size=self.rf_size[i],
                                      mapping_matrix=self.mapping_matrix[i]) for k in range(q[i])]

    def learn(self, image, lbl, layer, dim, red_dim=True):
        images = self.create_receptive_field(layer, image=image, rf_size=self.rf_size[layer], stride=self.stride[layer])
        self.weights[layer] = [self.stams[layer][k](images[k], lbl) for k in range(len(self.stams[layer]))]
        # if layer == 1:
        #         #     pd.set_option('display.max_rows', 90)
        #         #     pd.set_option('display.max_columns', 90)
        #         #     pd.set_option('display.width', 1000)
        #         #     print self.weights[layer]
        if red_dim:  # if are at the last layer, don't reduce dimension.
            self.weights[layer] = [self.reduce_dimension(self.weights[layer][k], dim=dim) for k in range(len(self.weights[layer]))]
            # assembled_image_vector = np.asarray(self.weights[layer]).reshape(1, len(self.weights[layer][0])*len(self.weights[layer]))
            dim = int(np.sqrt(len(self.weights[layer])))
            assembled_image_vector = self.assemble_image(self.weights[layer], dim=(dim, dim))
            dim = assembled_image_vector.shape[0]
            assembled_image_vector = assembled_image_vector.reshape(1, dim*dim)
            return assembled_image_vector

        elif (not red_dim) and (layer != self.layers-1):
            dim = int(np.sqrt(len(self.weights[layer])))
            assembled_image_vector = self.assemble_image(self.weights[layer], dim=(dim, dim))
            dim = assembled_image_vector.shape[0]
            assembled_image_vector = assembled_image_vector.reshape(1, dim * dim)
            return assembled_image_vector

        elif (not red_dim) and (layer == self.layers-1):
            pass  # pass for now

    def run(self, data, lbl):
        t=0
        while t < data.shape[0]:
            layer = 0
            if t > 1 and t%50 == 0:
                print '{} images completed!'.format(t)
            image = data.iloc[t, :].values.reshape(1, len(data.iloc[t, :]))
            while layer < self.layers:
                image = self.learn(image=image, lbl=lbl[t], layer=layer, dim=self.dim[layer], red_dim=self.red_dim[layer])
                layer += 1
            t+=1

    def get_weights(self, layer):
        result = (self.weights[layer - 1])[0]
        return result

    def create_receptive_field(self, layer, image, rf_size, stride):

        if stride is None:
            stride = 1
        store_rf = list()
        row = int(np.sqrt(image.shape[1]))
        image = image.reshape(row, row)
        track_row = 0
        image = pd.DataFrame(data=image)
        if image.shape[0] == rf_size:
            store_rf.append(image.values.reshape(row*row,))
            store_rf = np.asarray(store_rf)
            return store_rf
        while (track_row < image.shape[0] - rf_size + 1):
            track_col = 0
            while (track_col < image.shape[0] - rf_size + 1):
                rf = image.iloc[track_row:track_row + rf_size, track_col:track_col + rf_size]
                track_col += stride
                store_rf.append(list(rf.values.reshape(rf_size * rf_size,)))
            track_row += stride
        store_rf = np.asarray(store_rf)
        if len(store_rf) == 0:
            raise Hierarchy_STAM_Exception('no receptive field created\n'
                                           'please check the rf_size and stride for layer {}!'.format(layer+1))
        return np.array(store_rf)

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
        return temp

    def assemble_image(self, data, dim=(2, 5)):
        check_dim = [(data[0].shape == data[i].shape) + 0 for i in range(1, len(data))]
        if sum(check_dim) < len(data) - 1:
            raise (ValueError('attempt to concatenate incompatible data'))
        elif dim[0] * dim[1] != len(data):
            raise (ValueError('incompatible dimension'))
        each_row = {}
        count = 0
        i = 0
        for i in range(dim[0]):
            each_row[i] = data[count]
            count += 1
            j = 0
            for j in range(0, dim[1] - 1):
                temp = np.concatenate([each_row[i], data[count]], axis=1)
                each_row[i] = temp
                count += 1
                j += 1
            i += 1
        temp = each_row[0]
        for k in range(1, len(each_row)):
            temp = np.concatenate([temp, each_row[k]], axis=0)
        return temp


def main():
    data = Extract_Data()
    data, lbl = data.get_prep_data(type='train')
    data = data.head(100)
    lbl = lbl[:100]
    # dim signifies the dimension we want to reduce the RF to.
    hierarchy_stams = Hierarchy_STAMS(data, lbl,  stride=[2,  4, 1], layers=3, neurons=[20, 20, 10], dim=[4, 2, None], rf_size=[4, 26, 18],
                 red_dim=(True, True, False), mapping_matrix=(True, True, False))
    hierarchy_stams.run(data, lbl)
    W = hierarchy_stams.get_weights(2)
    for i in W.index.tolist():
        util.display_image(image=W.iloc[i, :], rows=24, cols=24)

if __name__ == '__main__':
    main()