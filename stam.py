#@author=Joseph_Aribido
import sys
import util
import pandas as pd
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import os
import copy


SAVE_MODEL_PATH = '/Users/joseph/Desktop/Dev/neural_algo/dataset/pickle_obj'

class STAM_Exception(Exception):
    'Returns and Exception for STAM'

class STAM:

    def __init__(self, num_train=40000, num_test=10000, num_valid=10000):
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
        self.cluster_labels = ['cluster A', 'cluster B', 'cluster C', 'cluster D', 'cluster E', 'cluster F', 'cluster G',
                               'cluster H', 'cluster k', 'cluster L', 'cluster M', 'cluster N', 'cluster O', 'cluster P',
                               'cluster Q']

    def get_data(self):
        return util.load()

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

    def balanced_order(self, data, lbl):
        print 'please wait, generating balanced order data...'
        build_indices = list()
        data['lbl'] = lbl
        data = data.sort_values('lbl') # sort based on the labels
        ten_value_index = [list(data['lbl']).index(x) for x in range(10)]
        ten_value_index = np.asarray(ten_value_index)
        indices = np.asarray(data.index.tolist())
        print 'building indices...'
        for i in range(int(data.shape[0]/10.0)):
            ten_value_index += 1
            temp = np.random.choice(a=indices[ten_value_index], size=len(ten_value_index), replace=False)
            build_indices.extend(temp)
        print 'balanced order indices generated successfully!'

        data = data.iloc[build_indices, :]
        print 'balanced order data created successfully...\n'
        return data

    def random_order(self, data, lbl):
        print 'please wait, generating random order data...'
        df = pd.DataFrame(data=data, columns=range(data.shape[1]))
        df['lbl'] = lbl
        df = df.sample(frac=1)
        return df


    def neural_algo(self, data,
                    lbl,
                    W=None,
                    num_clusters=10,
                    iterative_prediction=False,
                    get_mapping_matrix=False,
                    order='BALANCED_ORDER',
                    W_init = 'RND_INT'):

        matrix = pd.DataFrame(data=np.zeros([num_clusters, 10]), index=self.cluster_labels[0:num_clusters], dtype=int)
        # we hard code 10 to be the number of columns because there are 10
        # classes in the mnist dataset no matter what.
        # Initialize W, thetha and n
        indices_of_W = range(0, num_clusters)
        track_predictions = list()
        df = None
        if order == 'BALANCED_ORDER':
            df = self.balanced_order(data=data, lbl=lbl)

        elif order == 'RANDOM_ORDER':
            df = self.random_order(data=data, lbl=lbl)

        else:
            df = data
            df['lbl'] = lbl

        print 'proceeding with neural-algorithm...'
        if W is None:
            # We give this option in case we have some learned W for which we want to optimize with further training
            # This is useful if we want to train our algorithm in several epochs.
            if W_init == 'RND_INT':
                print 'initializing W with random digits...'
                W = pd.DataFrame(np.random.rand(num_clusters, df.shape[1]-1))
            elif W_init == 'OPT_INT':
                # Here, we initialize W with each digit class.
                print 'initializing W with each digit class'
                W = pd.DataFrame(np.zeros([num_clusters, df.shape[1]-1]))
                digits = list(np.random.choice(a=range(10), size=num_clusters, replace=False))
                while len(digits) > 0:
                    # We search over the labels for each digit and when we find it, we initialize W with it
                    j = 0
                    while df.iloc[df.index.values[j], -1] != digits[0]:
                        j += 1
                    else:
                        W.iloc[len(digits)-1, :] = df.iloc[df.index.values[j], :-1]
                        digits.pop(0)
            else:
                raise(STAM_Exception('please specify W_int=RND_INT or W_int=OPT_INT'))
        print 'W initialized...\n'

        thetha = np.zeros([W.shape[0], ])
        n_i = np.ones([W.shape[0], ])
        # We iterate over the examples T times. Where T is the length of each example.
        print 'iterating over examples...'

        for T in df.index.tolist():

            x_T = df.iloc[T, :-1]
            c, y_i_t, z_T = self.a_T(W, x_T, thetha)
            n_i[c] = n_i[c] + 1
            W.iloc[c, :] = W.iloc[c, :] + 1 / n_i[c] * (2 * x_T - W.iloc[c, :])
            thetha[c] = thetha[c] + 1 / n_i[c] * (z_T - thetha[c])

            if get_mapping_matrix:
                matrix.iloc[c, int(df.iloc[T, -1])] += 1

            if iterative_prediction:
                print 'iteractive prediction activated. Generating accuracy for each example...'
                # for iterative prediction we will have to stick to a sequential iteration of the data to track the
                # various predictions of W_i. However, to introduce some randomness, we suggest the data should be
                # randomized before feeding it into this algorithm.
                best_image_value = [util.ssim_two_images(W.iloc[i, :], data.iloc[T, :]) for i in indices_of_W]
                index = best_image_value.index(max(best_image_value))
                track_predictions.insert(len(track_predictions), (T, index))

        print 'neural algorithm completed! \n'
        if iterative_prediction:
            return 0.5 * W, track_predictions
        if get_mapping_matrix:
            return 0.5 * W, matrix
        return 0.5 * W, 0

    def a_T(self, W_T, X_T, thetha_T):
        a_i_T = -np.dot(W_T, X_T) + thetha_T
        c = np.argmin(a_i_T, axis=0)
        y_i_t = np.zeros_like(a_i_T)
        y_i_t[c] = 1
        z_T = -1*a_i_T[c]
        return c, y_i_t, z_T

    def run(self, data, lbl, W=None, num_clusters=10, order=None, get_iterative_prediction=False,
            get_mapping_matrix=False, W_init=None):

        if get_iterative_prediction:
            W, pred = self.neural_algo(data, lbl, W=W, num_clusters=num_clusters, iterative_prediction=get_iterative_prediction,
                                       get_mapping_matrix=get_mapping_matrix, order=order, W_init=W_init)
            W.to_pickle(SAVE_MODEL_PATH)
            return W, pred
        elif get_mapping_matrix:
            W, matrix = self.neural_algo(data, lbl, W=W, num_clusters=num_clusters, iterative_prediction=get_iterative_prediction,
                                       get_mapping_matrix=get_mapping_matrix, order=order, W_init=W_init)
            W.to_pickle(SAVE_MODEL_PATH)
            return W, matrix
        else:
            W.to_pickle(SAVE_MODEL_PATH)
            return W, 0

    def learn_labels(self, validation_data, validation_lbl, W):
        """This function attempts to assign labels to the various images in the converged W"""
        labels = [[] for _ in range(W.shape[0])]
        actual_labels = list()
        for i in range(validation_data.shape[0]):
            image_similarity = [util.ssim_two_images(W.iloc[j, :], validation_data.iloc[i, :]) for j in range(W.shape[0])]
            index_pos = image_similarity.index(max(image_similarity))
            labels[index_pos].append(validation_lbl[i])

        for each_label in labels:
            a = list(set(each_label))
            val = [each_label.count(x) for x in a]
            if val == []:
                actual_labels.append([])
            else:
                actual_labels.insert(len(actual_labels), a[val.index(max(val))])
        return actual_labels

    def predict(self, W, W_lbl, image, img_lbl):
        simm_compare = [util.ssim_two_images(W.iloc[i, :], image) for i in range(0, W.shape[0])]
        simm_pred = W_lbl[simm_compare.index(max(simm_compare))]
        self.simm_accuracy.append((simm_pred == img_lbl)+0)

    def print_W(self):
        print 'centroid print requested...'
        with open(SAVE_MODEL_PATH, 'rb') as f:
            W = pickle.load(f)
        for i in range(W.shape[0]):
            util.display_image(W.iloc[i, :], image_name=self.cluster_labels[i])


class Experiments:
    def __init__(self):
        self.stam_obj = STAM()

    def check_training_accuracy(self):
        test_data, test_lbl = self.stam_obj.get_prep_data(type='test')
        self.stam_obj.run(test_data, test_lbl)
        with open(SAVE_MODEL_PATH, 'rb') as f:
            W = pickle.load(f)
        W_lbl = [7, 4, 8, 3, 6, 9, 2, 5, 2, 0, 1, 9, 8, 6, 1]
        for T in range(test_data.shape[0]):
            # for each image in the test set, we will predict on them and check the accuracies of mse and simm
            self.stam_obj.predict(W, W_lbl, test_data.iloc[T, :], test_lbl[T])
        result1 = self.stam_obj.simm_accuracy
        print np.double(sum(result1)) / self.stam_obj.num_test
        result1 = self.moving_average(result1, 100)
        plt.plot(result1)
        plt.plot(np.unique(range(len(result1))),
                 np.poly1d(np.polyfit(range(len(result1)), result1, 4))(np.unique(range(len(result1)))), linewidth=2.5)
        plt.title('prediction accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('number of examples')
        plt.show()

    def learn_labels_after_training(self, train=False):
        if train:
            test_data, test_lbl = self.stam_obj.get_prep_data(type='test')
            self.stam_obj.run(test_data, test_lbl)
        with open(SAVE_MODEL_PATH, 'rb') as f:
            W = pickle.load(f)
        valid_img, valid_lbl = self.stam_obj.get_prep_data(type='validation')
        labels = self.stam_obj.learn_labels(valid_img, self.stam_obj.valid_lbl, W)
        print labels

    def track_errors_during_training(self, num_of_clusters):
        data, lbl = self.stam_obj.get_prep_data(type='train')
        W, track_predictions = self.stam_obj.neural_algo(data,
                                                         num_clusters=num_of_clusters,
                                                         iterative_prediction=True)
        W_lbl = [2, 1, 0, 7, 9, 8, 6, 5, 1, 3] # we learned this by running the learn_labels_after_training experiment
        errors =list()
        for i in range(len(track_predictions)):
            errors.append((self.stam_obj.train_lbl[track_predictions[i][0]] == W_lbl[track_predictions[i][1]]) + 0)
        result1 = self.moving_average(errors, 100)
        plt.plot(result1)
        plt.plot(np.unique(range(len(result1))), np.poly1d(np.polyfit(range(len(result1)), result1, 4))(np.unique(range(len(result1)))), linewidth=3)
        plt.title('tracking error during training')
        plt.ylabel('error')
        plt.xlabel('number of training examples')
        plt.show()

    def learn_w_with_set_clusters(self, data, lbl, num_clusters, shuffle=False, use_saved_W=False, num_epochs=0, get_mapping_matrix=False, order=None, W_init=None):
        matrix = None
        for i in range(num_epochs):
            if use_saved_W:
                if os.path.exists(SAVE_MODEL_PATH):
                    with open(SAVE_MODEL_PATH, 'rb') as f:
                        W = pickle.load(f)
                        _, matrix = self.stam_obj.run(data, lbl, W=W, num_clusters=num_clusters,
                                                      get_mapping_matrix=get_mapping_matrix, order=order, W_init=W_init)
                else:
                    raise(pickle.PickleError('An error occurred with the file path!'))
            else:
                _, matrix = self.stam_obj.run(data, lbl, num_clusters=num_clusters,
                                              get_mapping_matrix=get_mapping_matrix, order=order, W_init=W_init)
        return matrix

    def moving_average(self, values, window):
        weights = np.repeat(1.0, window) / window
        sma = np.convolve(values, weights, 'valid')
        return sma

    def run_experiments(self):
        train, train_lbl = self.stam_obj.get_prep_data(type='train')
        test, test_lbl = self.stam_obj.get_prep_data(type='test')
        num_clusters = 10
        # settings for DataFrame display
        pd.set_option('display.max_rows', 1000)
        pd.set_option('display.max_columns', 1000)
        pd.set_option('display.width', 1000)
        #self.learn_labels_after_training(train=False)
        #self.check_training_accuracy(train=False)
        #self.track_errors_during_training(num_of_clusters=15)
        matrix = self.learn_w_with_set_clusters(train, train_lbl, num_clusters=num_clusters, use_saved_W=False, num_epochs=1,
                                                get_mapping_matrix=True, order='BALANCED_ORDER', W_init='RND_INT')
        print matrix
        writer = pd.ExcelWriter('rnd_init_uni_seq.xlsx')
        matrix.to_excel(writer, 'Sheet1')
        writer.save()
        self.stam_obj.print_W()



def main():
    expt = Experiments()
    expt.run_experiments()

if __name__=='__main__':
    main()