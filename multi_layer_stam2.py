import numpy as np
import pandas as pd
import util
import matplotlib.pyplot as plt
from matplotlib import gridspec
from warnings import warn
import copy

class Hierarchy_STAM_Exception(Exception):
    "Defines errors related to Hierarchical_STAMs"


class STAM_Exception(Exception):
    "Defines STAM errors"


class STAM:

    def __init__(self, num_clusters, stam_id, rf_size, layer, init_images, init_lbls, init_W=None,
                 img_len=49, alpha=0.01, max_layer=1):
        self.num_clusters = num_clusters
        self.stam_id = stam_id
        self.rf_size = rf_size
        self.layer = layer
        self.init_w = init_W
        self.img_len = img_len
        self.alpha = alpha
        self.init_images = init_images
        self.init_lbls = init_lbls
        self.max_layer = max_layer
        self.W = self.initialize_W(init_type=self.init_w, clusters=self.num_clusters, images=init_images,
                                   label=init_lbls, img_len=self.img_len, layer=self.layer)
        self.check_parameters()

    def __call__(self, image, lbl, output_type):
        """
        The call function implements a custom call method to the STAM class to enable other algorithms call a particular
        STAM object
        :param image: This is the image being passed for clustering
        :param lbl:
        :return:
        """
        stam_output = self.stam(image, lbl, output_type)
        return stam_output

    def check_parameters(self):
        if self.W.shape[1] != self.img_len:
            print 'initialization images have length: {}'.format(self.init_images.shape[1])
            print 'calculated image length: {}'.format(self.img_len)
            raise(STAM_Exception('please check initialization image dimension'))

    def stam(self, image, lbl, output_type):
        """
        :param image: the image to be clustered
        :param lbl: the label of the image
        :return:
        """
        # print image.shape
        assert(image.shape == (1, max(image.shape)))
        (c_ind, c_dist, d_ind, d_dist) = self.distance(self.W, image)
        self.W[c_ind, :] = (1 - self.alpha) * self.W[c_ind, :] + self.alpha * image
        if output_type == 'one_centroid':
            return self.W[c_ind, :].reshape(1, self.W.shape[1])
        elif output_type == 'all_centroid':
            return self.W

    def initialize_W(self, init_type, clusters, images, label, img_len, layer=0):
        """
        This function initializes the centroid we want to use.
        :param init_type: specifies the initialization type.
        :param clusters: number of clusters in our dataset
        :param img_len: the length of each image vector
        :param images: the images present in our datasetgh jyn b,m
        :param label: the label of the images
        :param m: the
        :return: W
        """
        if init_type == 'opt':
            if layer < self.max_layer:
                W = self.initialize_receptive_field_clusters(images, label, clusters, img_len=img_len, layer=layer)
            else:
                W = self.initialize_averaged_clusters(images, label, clusters=clusters)
        elif init_type == 'random':
            W = np.random.rand(clusters, img_len)*0.01
        elif init_type == 'zero':
            W = np.zeros([clusters, img_len])
        return W

    def initialize_averaged_clusters(self, images, label, clusters):
        """
        This function creates the average of the 10 clusters
        :param images: image dataset
        :param label: labels of the images
        :param clusters: the number of clusters
        :param m: the number of averaged examples per cluster
        :return: the initialized centroids and the labels
        """
        label = np.array(label)
        m = int(images.shape[0]/clusters)
        W = np.zeros([clusters, images.shape[1]])
        label_count = [m for _ in range(clusters)]
        while sum(label_count) > 0:
            inds = np.random.choice(range(images.shape[0]), size=clusters, replace=False)
            selected_labels = label[inds]
            for i in range(len(inds)):
                if label_count[selected_labels[i]] > 0:
                    W[selected_labels[i], :] = W[selected_labels[i], :] + images[inds[i], :]
                    label_count[selected_labels[i]] -= 1
        W = W / m
        return W

    def initialize_receptive_field_clusters(self, init_images, init_labels, clusters, img_len, layer=0, seed=None):
        """
        Here we create layer 1 centroids using the receptive fields from the images
        :param W: the centroid of the stam to be initialized
        :param images: images from which to create receptive fields
        :param label:
        :param rf_size:
        :param clusters:
        :param m: the number of centroids imposed
        :return: W
        """
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(init_images)
        W = np.zeros([clusters, img_len])
        rf_size = int(np.sqrt(W.shape[1]))
        orignal_img_sz = int(np.sqrt(init_images.shape[1]))
        for i in range(W.shape[0]):
            # recall that receptive takes in an image that is shape (1, dimension of image)
            rf_images = Hierarchy_STAM.create_receptive_field(init_images[i, :].reshape(1, orignal_img_sz*orignal_img_sz),
                                                              rf_size=7, stride=3, layer=layer)
            if rf_images.shape[0] < W.shape[0]:
                raise STAM_Exception('incompatible initialization setup')
            W[i, :] = W[i, :] + rf_images[i, :]
        return W

    def distance(self, W, x_t):
        """
        This function calculates the smallest distances between each centroid in W and x_t
        :param W: matrix of centroids
        :param x_t: example image
        :return: the smallest two distances and their indices
        """
        distances = np.square(np.linalg.norm(W - x_t, ord=2, axis=1))
        # if self.opt is False:
        #     print distances
        indices = distances.argsort()[:2]
        c_ind, d_ind = indices[0], indices[1]
        c_dist, d_dist = distances[indices]
        return (int(c_ind), c_dist, int(d_ind), d_dist)

    def get_w(self, stam_id, layer):
        if layer == self.layer and stam_id == stam_id:
            return self.W
        else:
            raise STAM_Exception('layer or stam_id mismatch error!')

    def print_centroid(self, W, figure=0):
        """
        The function prints the centroid W
        :param W: The centroid
        :return:
        """
        row = int(np.sqrt(W.shape[1]))
        fig = plt.figure(figure)
        for i in range(W.shape[0]):
            plt.subplot(int(np.sqrt(W.shape[0])), int(np.sqrt(W.shape[0])), i+1)
            plt.imshow(W[i, :].reshape(row, row))
            plt.axis('off')
        plt.show()

class Hierarchy_STAM:

    def __init__(self, data, lbl, init_images, init_lbls, stride=[2, 1], layers=2, neurons=(5, 10), red_dim=(2, 1), rf_size=[4, 26],
                output=('one_centroid', 'all_centroid'), init_type=('opt', 'zero'), alpha=0.01):

        self.data = data
        self.count = 0
        self.lbl = lbl
        self.stride = stride
        self.layers = layers
        self.neurons = neurons
        self.num_neighbors = 4
        self.red_dim = red_dim
        self.weights = {}
        self.matrices = {}
        self.n_i = {}
        self.layer = 0
        self.init_type = init_type
        self.params = self.intilialize_paramters(init_images, init_lbls, neurons, stride, red_dim, rf_size, alpha, output, init_type)

    def intilialize_paramters(self, init_images, init_lbls, neurons, stride, red_dim, rf_size, alpha, output, init_type):
        """
        Here, we dynamically calculate the size of the image at each layer after we assemble the receptive
        fields together.
        :param init_images: images to initialize all STAMs
        :param init_lbls:
        :param neurons:
        :param stride:
        :param red_dim:
        :param rf_size:
        :param alpha:
        :param output:
        :return: a dictionary of parameters
        """
        #
        n_li = int(np.sqrt(self.data.shape[1]))  # 28 by 28 size images in layer 1
        num_stams = {}
        overlap = {}
        params = {}
        stams = {}
        i = 0
        while i < self.layers:
            if i == 0:
                num_stams[i] = int(((n_li-rf_size[i])/stride[i])+1)**2
                n_li = int(np.sqrt(((rf_size[i]**2)*num_stams[i])/(red_dim[i])**2))
                this_layer_overlap = rf_size[i] - stride[i]
                next_layer_overlap = int(this_layer_overlap / (red_dim[i]))
                overlap[i] = next_layer_overlap
                n_li = int(n_li - (np.sqrt(num_stams[i]) - 1) * overlap[i])
            elif i < self.layers-1:
                num_stams[i] = int(((n_li - rf_size[i]) / stride[i]) + 1) ** 2
                n_li = int(np.sqrt(((rf_size[i] ** 2) * num_stams[i]) / (red_dim[i]) ** 2))
                this_layer_overlap = rf_size[i] - stride[i]
                next_layer_overlap = int(this_layer_overlap / (red_dim[i]))
                overlap[i] = next_layer_overlap
                n_li = int(n_li - (np.sqrt(num_stams[i]) - 1) * overlap[i])
            else:
                num_stams[i] = 1
                overlap[i] = 0
                rf_size[i] = n_li
                stride[i] = 1
            i += 1
        # Print out some stats...
        print '====STAM config====', '\n'
        print 'number of receptive fields / STAMs in each layer: ', num_stams
        print 'number of overlaps in each regions: ', overlap
        print 'size of receptive fields in each layer: ', rf_size
        print 'number of strides per layer: ', stride
        print '\n'

        # initialize the STAM modules...
        for i in range(self.layers):
            stams[i] = [STAM(num_clusters=neurons[i], stam_id=k, rf_size=rf_size[i],
                                  layer=i, init_images=init_images, init_lbls=init_lbls, init_W=init_type[i],
                                  img_len=rf_size[i]*rf_size[i], alpha=alpha) for k in range(num_stams[i])]
            params['neurons'] = neurons
            params['stride'] = stride
            params['red_dim'] = red_dim
            params['rf_size'] = rf_size
            params['alpha'] = alpha
            params['num_stams'] = num_stams
            params['output'] = output
            params['overlap'] = overlap
            params['stams'] = stams
        return params

    def run(self, data, lbl):
        t = 0
        while t < data.shape[0]:
            layer = 0
            if t > 1 and t%50 == 0:
                print '{} images completed!'.format(t)
            #image = data.iloc[t, :].values.reshape(1, len(data.iloc[t, :]))
            image = data[t, :].reshape(1, data.shape[1])
            while layer < self.layers:
                image = self.learn(image=image, lbl=int(lbl[t]), layer=layer)
                layer += 1
            t += 1

    def learn(self, image, lbl, layer):
        """
        This function implements the learning portion of the hierarchical STAM by clustering modules in an unsupervised
        manner.
        :param image: the image to learn
        :param lbl: the image label
        :param layer: the layer we are implementing learning.
        :param dim:
        :return: an assembled image if this is not the last layer.
        """
        assert(image.shape == (1, max(image.shape)))
        images = self.create_receptive_field(image=image, rf_size=self.params['rf_size'][layer],
                                             stride=self.params['stride'][layer], layer=layer)

        # the list comprehension below we are making a call to each stam in the self.params['stams'] instance. Note
        # each stam is a different instance of a general STAM module. the __call__() function allows us to call the
        # STAM instance as stam(image=image, lbl=label, output_type='output_type')

        self.weights[layer] = [self.params['stams'][layer][k](image=images[k, :].reshape(1, images.shape[1]),
                                                              lbl=lbl, output_type=self.params['output'][layer])
                               for k in range(len(self.params['stams'][layer]))]

        # reshape the list of images in self.weights[layer] into an N by M matrix
        array_length = len(self.weights[layer])
        one_arr_length = self.weights[layer][0].shape[1]
        self.weights[layer] = np.array(self.weights[layer]).reshape(array_length, one_arr_length)

        if layer < self.layers-1:
            if self.params['red_dim'][layer] > 1:
                self.weights[layer] = self.reduce_dimension(self.weights[layer], dim=self.params['red_dim'][layer],
                                                             type='average')
            if self.params['overlap'][layer] > 0:
                image_vector = self.squeeze_image(images=self.weights[layer], shift=self.params['overlap'][layer])
            else:
                image_vector = self.assemble_imae(data=self.weights[layer], sep=abs(self.params['overlap'][layer]))
            return image_vector
            # todo: Feedback to be implemented here...
        else:
            image_vector = self.assemble_image(data=self.weights[layer], sep=self.params['overlap'][layer])
            return image_vector
            # we do not return the image in the last layer because this is the last step in the heirarchy.

    def get_stams(self, layer):
        return self.params['stams'][layer-1]

    def reduce_dimension(self, image, dim=2, type='ave', seed=None, print_det=False):
        """
        This function reduces the size of an image by a factor of dim
        :param image: image of shape: 1, image_length
        :param dim: factor by which image is to be reduced
        :param type: average pooling or max pooling
        :return: reduced image
        """
        if seed is not None:
            np.random.seed(seed)
        # create an empty pd of the desired DataFrame

        actual_dim = int(np.sqrt(image.shape[1]))
        desired_dim = int(actual_dim / dim)
        new_image = np.zeros(image.shape[0], desired_dim*desired_dim)
        if print_det:
            print 'actual_dim: ', actual_dim
            print 'desired_dim: ', desired_dim

        if actual_dim % dim != 0:
            warning = ('attempting to resize image of size {} '
                       'to size {}').format(actual_dim, desired_dim)
            warn(warning)
        for m in image.shape[0]:
            temp = np.zeros([desired_dim, desired_dim])
            new_data = image[m, :].reshape(actual_dim, actual_dim).astype(np.float32)
            for j in range(desired_dim):
                for k in range(desired_dim):
                    if type=='ave':
                        print [new_data[j*dim: j*dim+dim, k*dim: k*dim+dim]]
                        temp[j, k] = np.mean(new_data[j*dim: j*dim+dim, k*dim: k*dim+dim])
                    elif type=='max':
                        temp[j, k] = np.max(new_data[j * dim: j * dim + dim, k * dim: k * dim + dim])
            new_image[m, :] = temp.reshape(1, temp.shape[0]*temp.shape[1])
        return new_image

    def assemble_image(self, data, sep):
        """
        This function assemebles image pieces in a numpy array where each row is an image
        :param data: a list of numpy arrays. The numpy array is of shape (n, rf*rf). each row is a separate imafe
        :param sep: the amount of separation between the images as an integer
        :return: the assembled image as a numpy array of shape (N, N)
        """
        len_one_row = data.shape[1]
        shape_one_row = int(np.sqrt(len_one_row))
        dim = int(np.sqrt(data.shape[0]))
        new_data = [data[i, :].reshape(shape_one_row, shape_one_row) for i in range(data.shape[0])]
        data = new_data
        check_dim = [(data[0].shape == data[i].shape) + 0 for i in range(1, len(data))]
        dim = (dim, dim)
        if sum(check_dim) < len(data) - 1:
            raise (ValueError('attempt to concatenate incompatible data failed'))
        elif dim[0] * dim[1] != len(data):
            raise (ValueError('incompatible dimension'))
        each_row = {}
        count = 0
        for i in range(dim[0]):
            each_row[i] = data[count]
            count += 1
            if sep > 0:
                pad = np.zeros([data[0].shape[0], sep])
                for j in range(0, dim[1] - 1):
                    temp = np.concatenate([each_row[i], pad], axis=1)
                    temp = np.concatenate([temp, data[count]], axis=1)
                    each_row[i] = temp
                    count += 1
            else:
                for j in range(0, dim[1] - 1):
                    temp = np.concatenate([each_row[i], data[count]], axis=1)
                    each_row[i] = temp
                    count += 1
        temp = each_row[0]
        for k in range(1, len(each_row)):
            if sep > 0:
                pad = np.zeros([sep, each_row[0].shape[1]])
                temp = np.concatenate([temp, pad], axis=0)
                temp = np.concatenate([temp, each_row[k]], axis=0)
            else:
                temp = np.concatenate([temp, each_row[k]], axis=0)
        return temp

    @classmethod
    def create_receptive_field(self, image, rf_size, stride, layer=1):
        """
        This function creates receptive fields for a given image.
        It converts each receptive field into a horizontal vector and stacks
        them into a matrix.
        :param image: the image to be broken into receptive fields
        :param rf_size: the length of the stride
        :param stride:
        :return: a numpy array of receptive fields extracted from an image
        """
        assert(image.shape == (1, image.shape[1]))
        if stride is None:
            stride = 1
        store_rf = list()
        row = int(np.sqrt(image.shape[1]))
        image = image.reshape(row, row)
        track_row = 0
        image = pd.DataFrame(data=image)
        while (track_row < image.shape[0] - rf_size + 1):
            track_col = 0
            while (track_col < image.shape[0] - rf_size + 1):
                rf = image.iloc[track_row:track_row + rf_size, track_col:track_col + rf_size]
                track_col += stride
                store_rf.append(list(rf.values.reshape(rf_size * rf_size, )))
            track_row += stride
        store_rf = np.asarray(store_rf)
        if len(store_rf) == 0:
            raise Hierarchy_STAM_Exception('no receptive field created\n'
                                           'please check the rf_size and stride for layer {}!'.format(layer + 1))
        return np.array(store_rf)

    def squeeze_image(self, images, shift):
        """
        This function squeeze the image to have an overlap of size shift.
        :param images: the image we want to squeeze
        :param shift: the measure of squeezing we desire
        :return: the squeezed image
        """
        len_one_row = images.shape[1]
        shape_one_row = int(np.sqrt(len_one_row))
        new_images = [images[i, :].reshape(shape_one_row, shape_one_row) for i in range(images.shape[0])]
        images = new_images
        num_row_blocks = int(np.sqrt(len(images)))
        keep_one_row = {}
        count = 0
        c = images[count].shape[1] - shift
        for i in range(0, num_row_blocks):
            keep_one_row[i] = images[count]
            count += 1
            for j in range(0, num_row_blocks-1):
                a = keep_one_row[i][:, -1*shift:]
                b = images[count][:, :shift]
                c = images[count].shape[1] - shift
                value = (a+b)/2
                keep_one_row[i][:, -1*shift:] = value
                keep_one_row[i] = np.concatenate([keep_one_row[i], images[count][:, -1*c:]], axis=1)
                count += 1
        temp2 = keep_one_row[0]
        for j in range(1, len(keep_one_row)):
            val = temp2[-1 * shift:, :] + keep_one_row[j][0:shift, :]
            val = val / 2
            temp2[-1 * shift:, :] = val
            temp2 = np.concatenate([temp2, keep_one_row[j][-1 * c:, :]], axis=0)
        return temp2



def test_red_dimension():

    np.random.seed(4)
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    images, lbl, _, _ = util.Data_Processing(num_train=200, preprocessing='scaled').run()
    images, lbl = util.Data_Processing.random_order(images, lbl)
    init_images, init_lbls = util.Data_Processing.extract_unique_m_images(images=images[0:150, :], label=lbl[:150], m=6)
    images = images[150:200, :]
    lbl = lbl[150:200]
    stride = [3, 1]
    layers = 2
    neurons = (40, 10)
    red_dim = (1, 1)
    rf_size = [7, 28]
    output = ('one_centroid', 'all_centroid')
    init_type = ('opt', 'opt')
    alpha = 0.01
    h_stam = Hierarchy_STAM(data=images, lbl=lbl, init_images=init_images, init_lbls=init_lbls, stride=stride, layers=2,
                            neurons=neurons, red_dim=red_dim, rf_size=rf_size, output=output,
                            init_type=init_type, alpha=alpha)
    # stam = STAM(num_clusters=neurons[0], stam_id=0, rf_size=rf_size[0],
    #      layer=0, init_images=init_images,  init_lbls=init_lbls, init_W=init_type[0], img_len=rf_size[0]*rf_size[0],
    #      alpha=alpha)
    #c = stam(images[0, 300:349].reshape(1, 49), 1, output[0])
    plt.figure(0)
    print images.shape
    plt.imshow(images[0, :].reshape(28, 28))
    plt.show()
    plt.figure(1)
    squeezed_image = h_stam.learn(image=images[0, :].reshape(1, 784), lbl=lbl[0], layer=0)
    plt.imshow(squeezed_image)
    plt.show()


test_red_dimension()