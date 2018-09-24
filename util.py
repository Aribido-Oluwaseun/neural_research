#@author=Joseph_Aribido
import os
import cPickle as pickle
from mnist import MNIST
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

SAVE_IMAGE_PATH = '/Users/joseph/Desktop/Dev/neural_algo/imgs/covariance.xlsx'
PATH = '/Users/joseph/Desktop/Dev/neural_algo/dataset/mnist'
CIFAR = '/Users/joseph/Desktop/Dev/neural_algo/dataset/cifar-10-batches-py'

class Data_Processing_Error(Exception):
    "Defines data processing errors"


class Data_Processing:

    def __init__(self, num_train=10, num_test=10, num_valid=10, data_type='mnist', preprocessing='ZCA'):
        self.num_train = num_train
        self.num_test = num_test
        self.num_valid = num_valid
        self.data_type = data_type
        self.prep = preprocessing
        train_img, train_lbl, test_img, test_lbl = self.get_data(data_type=self.data_type, preprocessing=self.prep)
        self.valid_img = train_img[self.num_train: self.num_train + self.num_valid]
        self.valid_lbl = train_lbl[self.num_train: self.num_train + self.num_valid]
        self.train_img = train_img[0: self.num_train]
        self.train_lbl = train_lbl[0: self.num_train]
        self.test_img = test_img[0: self.num_test]
        self.test_lbl = test_lbl[0: self.num_test]

    def run(self):
        return self.train_img, self.train_lbl, self.test_img, self.test_lbl

    @staticmethod
    def random_order(data, lbl, seed=None):
        if seed is not None:
            np.random.seed(seed)
        print 'please wait, generating random order data...'
        lbl = lbl.reshape(len(lbl), 1)
        data = np.append(data, lbl, axis=1)
        np.random.shuffle(data)
        return data[:, :-1], data[:, -1].astype(np.int)

    @staticmethod
    def balanced_order(data, lbl, seed=None):
        if seed is not None:
            np.random.seed(seed)
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
                    a=range(len(data_holder[j])), size=1, replace=False)[0]]
                df.iloc[count, -1] = int(j)
                count += 1
            data_size -= 10
        data = df
        data = data.reset_index(drop=True)
        print 'balanced order data generated!\n'
        return data.values[:, :-1], data.values[:, -1].astype(np.int)

    @staticmethod
    def extract_unique_m_images(images, label, m, clusters=10):
        """
        This function extracts unique images from the random images and labels passed in.
        :param images:
        :param label:
        :param m:
        :param clusters:
        :return: a numpy array of images and their labels
        """
        store_images = np.zeros([clusters*m, images.shape[1]])
        store_lbls = list()
        label_count = [m for _ in range(clusters)]
        count = 0
        while sum(label_count) > 0:
            inds = np.random.choice(range(images.shape[0]), size=clusters, replace=False)
            selected_lbls = label[inds]
            for i in range(len(inds)):
                if label_count[selected_lbls[i]] > 0:
                    store_images[count, :] = images[inds[i], :]
                    store_lbls.append(selected_lbls[i])
                    count += 1
                    label_count[selected_lbls[i]] -= 1
        return store_images, store_lbls

    def load_cifar_names(self):
        filename = os.path.join(CIFAR, 'batches.meta')
        names = pickle.load(open(filename, 'rb'))
        return names['label_names']


    def load_data(self, type_data='mnist'):
        if type_data == 'mnist':
            mndata = MNIST(PATH)
            train_img, train_lbl = mndata.load_training()
            test_img, test_lbl = mndata.load_testing()
            # note that the image files contain values of 0-255. We should normalize the image files for better performance
            return np.asarray(train_img).astype(np.double), np.array(train_lbl), np.asarray(test_img).astype(np.double), np.asarray(test_lbl)

        elif type_data == 'cifar':
            all_files = os.listdir(CIFAR)
            test_file = os.path.join(CIFAR, 'test_batch')
            indices = [k for k in range(len(all_files)) if all_files[k].startswith('data_batch')]
            all_files = [all_files[k] for k in indices]
            data = np.empty([10000 * len(all_files), 32 * 32])
            test_data = np.empty([10000, 32 * 32])
            label = []
            count = 0
            file = [os.path.join(CIFAR, all_files[k]) for k in range(len(all_files))]
            for i in range(len(file)):
                with open(file[i], 'rb') as fo:
                    dict = pickle.load(fo)
                    img = dict['data']
                    lbl = dict['labels']
                    label.extend(lbl)
                    for z in range(img.shape[0]):
                        temp = np.transpose(np.reshape(img[z], (3, 32, 32)),
                                            (1, 2, 0))  # change the image into shape 32 by 32 by 3
                        temp = self.rgb2gray(temp)
                        data[count, :] = temp.reshape(32 * 32, )
                        count += 1
            count = 0
            with open(test_file, 'rb') as fo:
                dict = pickle.load(fo)
                img = dict['data']
                lbl = dict['labels']
                for z in range(img.shape[0]):
                    temp = np.transpose(np.reshape(img[z], (3, 32, 32)),
                                        (1, 2, 0))  # change the image into shape 32 by 32 by 3
                    temp = self.rgb2gray(temp)
                    test_data[count, :] = temp.reshape(32 * 32, )
                    count += 1
            train_img = data.astype(np.double)
            train_lbl = np.asarray(label)
            test_img = test_data.astype(np.double)
            test_lbl = np.asarray(lbl)
            return train_img, train_lbl, test_img, test_lbl
        else:
            raise(Data_Processing_Error('invalid data type specified!'))

    def load_scaled_data(self, type_data='mnist'):
            train_img, train_lbl, test_img, test_lbl = self.load_data(type_data)
            return train_img.astype(np.double)/255, np.asarray(train_lbl), test_img.astype(np.double)/255, np.asarray(test_lbl)

    def normalize_data(self, type_data='mnist'):
        train_img = None
        test_img = None
        train_lbl = None
        test_lbl = None

        train_img, train_lbl, test_img, test_lbl = self.load_data(type_data)
        train_img, test_img = np.array(train_img).astype(np.double), np.array(test_img).astype(np.double)
        train_img = train_img - np.mean(train_img, axis=1).reshape(train_img.shape[0], 1)
        test_img = test_img - np.mean(test_img, axis=1).reshape(test_img.shape[0], 1)
        train_img = (train_img / np.sqrt(np.var(train_img, axis=1) + 10).reshape(train_img.shape[0], 1))
        test_img = (test_img / np.sqrt(np.var(test_img, axis=1) + 10).reshape(test_img.shape[0], 1))
        pd.set_option('display.max_columns', 1000)
        return train_img, train_lbl, test_img, test_lbl

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def zca_keras(self, type_data='mnist'):
        batch = 100
        image_len = 0
        X_train, y_train, X_test, y_test = self.load_data(type_data=type_data)
        train_img = np.zeros_like(X_train)
        test_img = np.zeros_like(X_test)
        train_lbl = np.zeros_like(y_train)
        test_lbl = np.zeros_like(y_test)
        K.set_image_dim_ordering('th')
        # load data
        # reshape to be [samples][pixels][width][height]
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)
        if type_data == 'mnist':
            X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
            X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
            image_len = 784
        elif type_data == 'cifar':
            X_train = X_train.reshape(X_train.shape[0], 1, 32, 32)
            X_test = X_test.reshape(X_test.shape[0], 1, 32, 32)
            image_len = 1024
        # convert from int to float
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        # define data preparation
        #datagen = ImageDataGenerator(featurewise_center=True)
        datagen = ImageDataGenerator(zca_whitening=True)
        datagen.fit(X_train)
        count = 0
        stop_count = train_img.shape[0]

        for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=batch):
            if count == stop_count:
                break
            train_img[count: count+batch, :] = X_batch.reshape(batch, image_len)
            train_lbl[count: count+batch] = y_batch
            count += batch
        count = 0
        stop_count = test_img.shape[0]
        for X_batch, y_batch in datagen.flow(X_test, y_test, batch_size=batch):
            if count == stop_count:
                break
            test_img[count: count+batch, :] = X_batch.reshape(batch, image_len)
            test_lbl[count: count+batch] = y_batch
            count += batch
        return train_img, train_lbl, test_img, test_lbl


    def zca_whitening_matrix(self, X, print_cov=False):
        """
        ZCA function pulled from https://stackoverflow.com/questions/31528800/how-to-implement-zca-whitening-python
        """
        # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
        sigma_x = np.cov(X, rowvar=True) # [M x M]

        # Singular Value Decomposition. X = U * np.diag(S) * V
        print 'Generating whitening matrix...'
        Ux,Sx,Vx = np.linalg.svd(sigma_x)
            # U: [M x M] eigenvectors of sigma.
            # S: [M x 1] eigenvalues of sigma.
            # V: [M x M] transpose of U
        # Whitening constant: prevents division by zero
        epsilon = 1e-5
        # ZCA Whitening matrix: U * Lambda * U'
        ZCAx = np.dot(Ux, np.dot(np.diag(1.0/np.sqrt(Sx + epsilon)), Ux.T)) # [M x M]
        print 'Whitening examples... '
        ZCAx = np.dot(ZCAx, X)
        if print_cov:
            cov = np.around(a=np.cov(X), decimals=2)
            writer = pd.ExcelWriter(SAVE_IMAGE_PATH)
            pd.DataFrame(cov).to_excel(writer, 'Sheet1')
            writer.save()
        print '\n', 'data has been whitened!'
        return ZCAx


    def display_image(self, image, image_name='', save=False):
        row = int(np.sqrt(len(image)))
        plt.imshow(np.asarray(image).reshape(row, row), interpolation='bilinear')
        plt.title(image_name)
        if save:
            plt.savefig(image_name)
        else:
            plt.show()


    def display_digits(self, image, rows=28, cols=28):
        pd.set_option('display.height', 1000)
        pd.set_option('display.max_rows', 1000)
        pd.set_option('display.max_columns', 1000)
        pd.set_option('display.width', 1000)
        values = np.asarray(image).reshape(rows, cols)
        print (pd.DataFrame(values))


    def mse_two_images(self, image1, image2, rows=28, cols=28):
        image1 = np.asarray(image1).astype(np.double).reshape(rows, cols)
        image2 = np.asarray(image2).astype(np.double).reshape(rows, cols)
        mse_err = 1/(cols*rows)*np.sum(np.square(image1 - image2))
        return mse_err


    def get_data(self, data_type, preprocessing):
        if preprocessing == 'raw':
            return self.load_data(data_type)
        elif preprocessing == 'scaled':
            return self.load_scaled_data(data_type)
        elif preprocessing == 'normalized':
            return self.normalize_data(data_type)
        elif preprocessing == 'ZCA':
            return self.zca_keras(type_data=data_type)
        else:
            raise(Exception('data type or preprocessing not specified correctly...'))