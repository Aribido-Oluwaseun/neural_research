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

    def __init__(self, num_train=500, num_test=1000, num_valid=1000):
        self.num_train = num_train
        self.num_test = num_test
        self.num_valid = num_valid
        train_img, train_lbl, test_img, test_lbl = self.get_data(normalized=False)

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

    def get_data(self, normalized=False):
        if not normalized:
            return util.load()
        else:
            return util.normalize_data()

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

    def __init__(self, num_clusters, stam_id, rf_size, best_centroid=True, W_init='random'):
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
        self.best_centroid = best_centroid

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

        self.matrix.iloc[c, lbl] += 1
        # Here we attempt to get the nth best cluster using the mapping matrix
        #if self.mapping_matrix:
            # self.best_cluster = self.matrix[lbl].nlargest(n=self.num_clusters, keep='first')
            # self.best_cluster = self.best_cluster.index.tolist()[accuracy]
            # self.best_cluster = self.W.loc[self.best_cluster]
        if self.best_centroid:
            return self.W.loc[c]
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

    def __init__(self, data, lbl,  stride=(2, 1), layers=2, neurons=(5, 10), red_dim=(2, 1), rf_size=(4, 26),
                best_centroid=(True, False)):
        self.data = data
        self.lbl = lbl
        self.rf_size = rf_size
        self.stride = stride
        self.layers = layers
        self.neurons = neurons
        self.num_neighbors = 4
        self.red_dim = red_dim
        self.best_centroid = best_centroid
        self.weights = {}
        self.matrices = {}
        self.stams = {}
        self.n_i = {}
        self.layer = 0
        self.output_centroids = None
        self.overlap = {}
        self.intilialize_paramters()

    def intilialize_paramters(self):
        n_li = int(np.sqrt(len(self.data.iloc[0, :])))  # 28 by 28 size images in layer 1
        self.q = {}
        rf = {}
        rf[0] = self.rf_size[0]
        i = 0

        while i < self.layers:

            if i == 0:
                self.q[i] = int(((n_li-self.rf_size[i])/self.stride[i])+1)**2
                n_li = int(np.sqrt(((self.rf_size[i]**2)*self.q[i])/(self.red_dim[i])**2))
                this_layer_overlap = self.rf_size[i] - self.stride[i]
                next_layer_overlap = int(this_layer_overlap / (self.red_dim[i]))
                self.overlap[i] = next_layer_overlap
                n_li = int(n_li - (np.sqrt(self.q[i]) - 1) * self.overlap[i])

            elif i < self.layers-1:
                self.q[i] = int(((n_li - self.rf_size[i]) / self.stride[i]) + 1) ** 2
                n_li = int(np.sqrt(((self.rf_size[i] ** 2) * self.q[i]) / (self.red_dim[i]) ** 2))
                this_layer_overlap = self.rf_size[i] - self.stride[i]
                next_layer_overlap = int(this_layer_overlap / (self.red_dim[i]))
                self.overlap[i] = next_layer_overlap
                n_li = int(n_li - (np.sqrt(self.q[i]) - 1) * self.overlap[i])

            else:
                self.q[i] = 1
                self.overlap[i] = 0
                self.rf_size[i] = n_li
                self.stride[i] = 1

            i += 1
        print 'number of receptive fields in each layer: ', self.q
        print 'number of overlaps in each regions: ', self.overlap
        print 'size of receptive fields in each layer: ', self.rf_size
        for i in range(self.layers):
            self.stams[i] = [Stam(num_clusters=self.neurons[i], stam_id=k, rf_size=self.rf_size[i],
                                  best_centroid=self.best_centroid[i]) for k in range(self.q[i])]



    def learn(self, image, lbl, layer, dim):
        images = self.create_receptive_field(layer, image=image, rf_size=self.rf_size[layer], stride=self.stride[layer])

        self.weights[layer] = [self.stams[layer][k](images[k], lbl) for k in range(len(self.stams[layer]))]

        if layer < self.layers-1:
            current_dim = int(np.sqrt(np.array(self.weights[layer][0].shape[0])))
            reduced_dim = current_dim/dim
            self.weights[layer] = [self.reduce_dimension(self.weights[layer][k], dim=reduced_dim) for k in range(len(self.weights[layer]))]
            if self.overlap[0] == 0:
                size = int(np.sqrt(len(self.weights[layer])))
                assembled_image_vector = self.assemble_image(self.weights[layer], dim=(size, size))
            else:
                assembled_image_vector = self.squeeze_images(self.weights[layer], self.overlap[layer])
            dim = assembled_image_vector.shape[0]
            assembled_image_vector = assembled_image_vector.reshape(1, dim*dim)
            return assembled_image_vector



    def run(self, data, lbl):
        t = 0
        while t < data.shape[0]:
            layer = 0
            if t > 1 and t%50 == 0:
                print '{} images completed!'.format(t)
            image = data.iloc[t, :].values.reshape(1, len(data.iloc[t, :]))
            while layer < self.layers:
                image = self.learn(image=image, lbl=lbl[t], layer=layer, dim=self.red_dim[layer])
                layer += 1
            t += 1

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
        for i in range(dim[0]):
            each_row[i] = data[count]
            count += 1
            for j in range(0, dim[1] - 1):
                temp = np.concatenate([each_row[i], data[count]], axis=1)
                each_row[i] = temp
                count += 1
        temp = each_row[0]
        for k in range(1, len(each_row)):
            temp = np.concatenate([temp, each_row[k]], axis=0)
        return temp

    def squeeze_images(self, images, shift_size):
        initial_size = int(np.sqrt(len(images)))
        image_size = images[0].shape[0]
        temp = {}
        count = 0
        for i in range(initial_size):
            temp[i] = images[count]
            count += 1
            for j in range(0, initial_size - 1):
                a = image_size - shift_size
                val = temp[i][:, -1 * shift_size:] + images[count][:, 0:shift_size]
                val = val / 2
                temp[i][:, -1 * shift_size:] = val
                temp[i] = np.concatenate([temp[i], images[count][:, -1 * a:]], axis=1)
                count += 1
        temp2 = temp[0]
        for j in range(1, len(temp)):
            b = image_size - shift_size
            val = temp2[-1 * shift_size:, :] + temp[j][0:shift_size, :]
            val = val / 2
            temp2[-1 * shift_size:, :] = val
            temp2 = np.concatenate([temp2, temp[j][-1 * b:, :]], axis=0)
        return np.array(temp2)

    def get_weights(self, stam_id, layer):
        layer = layer-1
        return self.stams[layer][stam_id].get_W()

    def get_matrix(self, stam_id, layer):
        layer = layer-1
        return self.stams[layer][stam_id].get_matrix()

    def get_plot(self, stam_id, layer):
        matrix = self.get_matrix(stam_id, layer)
        total = matrix.sum(axis=1)
        total_sum = sum(total)
        total = total/total_sum
        plt.bar(x=range(len(total)), height=total)
        plt.title('weights in different clusters: layer: {}, stam_id: {}, total_examples: {}'.format(layer+1, stam_id, total_sum))
        plt.xlabel('examples per cluster')
        plt.ylabel('weights')
        name = str(stam_id)
        plt.savefig(name)


def main():
    data = Extract_Data()
    data, lbl = data.get_prep_data(type='train')
    layers = 2
    # dim signifies the dimension we want to reduce the RF to.
    hierarchy_stams = Hierarchy_STAMS(data, lbl,  stride=[2, 2, 0], layers=2, neurons=[20, 10, 10], red_dim=[1, 1, 1], rf_size=[2, 4, 4],
                 best_centroid=(True, True, False))
    hierarchy_stams.run(data, lbl)
    W = hierarchy_stams.get_weights(0, 2)
    for j in W.index.tolist():
        util.display_image(image=W.iloc[j, :].values, save=False)
    name = 'matrix' + '_' + str(0) + '_' + '10000.xlsx'
    writer = pd.ExcelWriter(name)
    M = hierarchy_stams.get_matrix(0, 2)
    M.to_excel(writer, 'Sheet1')
    writer.save()

    # values = [50, 60, 70]
    # for i in values:
    #     W = hierarchy_stams.get_weights(i, 1)
    #     M = hierarchy_stams.get_matrix(i, 1)
    #     name = 'matrix' + '_' + str(i) + '_' + '1000.xlsx'
    #     writer = pd.ExcelWriter(name)
    #     M.to_excel(writer, 'Sheet1')
    #     writer.save()
    #     hierarchy_stams.get_plot(i, 1)
    #     for j in W.index.tolist():
    #         name = str(i) + '_' + str(j)
    #         util.display_image(image=W.iloc[j, :].values, image_name=name, save=True)

if __name__ == '__main__':
    main()