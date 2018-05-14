# @author=Joseph_Aribido
import sys
import util
import pandas as pd
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt


SAVE_MODEL_PATH = '/home/ribex/Desktop/Dev/neural_algo/dataset/pickle_obj'


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

    def get_data(self):
        return util.load()

    def get_sme_accuracy(self):
        return self.mse_accuracy

    def get_simm_accuracy(self):
        return self.simm_accuracy

    def test(self):
        print self.test_lbl

    def make_df(self, data):
        df = pd.DataFrame(data=data, index=range(len(data)))
        return df

    def neural_algo(self, data, num_clusters=10, iterative_prediction=False, W=None):
        # Initialize W, thetha and n
        indices_of_W = range(0, num_clusters)
        track_predictions = list()
        if not W:
            # We give this option in case we have some learned W for which we want to optimize with further training
            # This is useful if we want to train our algorithm in several epochs.
            W = pd.DataFrame(np.zeros([num_clusters, data.shape[1]]))
        thetha = np.zeros([W.shape[0], ])
        n_i = np.ones([W.shape[0], ])
        # We iterate over T times
        for T in range(0, data.shape[0]):
            x_T = data.iloc[T, :]
            c, y_i_t, z_T = self.a_T(W, x_T, thetha)
            n_i[c] = n_i[c] + 1
            W.iloc[c, :] = W.iloc[c, :] + 1 / n_i[c] * (2 * x_T - W.iloc[c, :])
            thetha[c] = thetha[c] + 1 / n_i[c] * (z_T - thetha[c])
            if iterative_prediction:
                # for iterative prediction we will have to stick to a sequential iteration of the data to track the
                # various predictions of W_i. However, to introduce some randomness, we suggest the data should be
                # randomized before feeding it into this algorithm.
                best_image_value = [util.ssim_two_images(W.iloc[i, :], data.iloc[T, :]) for i in indices_of_W]
                index = best_image_value.index(max(best_image_value))
                track_predictions.insert(len(track_predictions), (T, index))
        if iterative_prediction:
            return 0.5*W, track_predictions
        return 0.5 * W

    def a_T(self, W_T, X_T, thetha_T):
        a_i_T = -np.dot(W_T, X_T) + thetha_T
        c = np.argmin(a_i_T, axis=0)
        y_i_t = np.zeros_like(a_i_T)
        y_i_t[c] = 1
        z_T = -1*a_i_T[c]
        return c, y_i_t, z_T

    def run(self, num_clusters=10, display=None):
        pickle_obj = None
        df = self.make_df(self.train_img)
        W = self.neural_algo(df, num_clusters, iterative_prediction=False)
        if display:
            for i in range(num_clusters):
                util.display_image(np.asarray(W.iloc[i, :]))
        open(SAVE_MODEL_PATH, 'w').close() #create a new file each time
        W.to_pickle(SAVE_MODEL_PATH)
        return True

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
        #mse_compare = [util.mse_two_images(W.iloc[i, :], image) for i in range(0, W.shape[0])]
        simm_compare = [util.ssim_two_images(W.iloc[i, :], image) for i in range(0, W.shape[0])]
        #mse_pred = W_lbl[mse_compare.index(min(mse_compare))]
        simm_pred = W_lbl[simm_compare.index(max(simm_compare))]
        # print img_lbl
        # print 'predicted simm index: ', simm_pred
        # print 'predicted sme index: ', mse_pred
        self.simm_accuracy.append((simm_pred == img_lbl)+0)

    def print_W(self):
        with open(SAVE_MODEL_PATH, 'rb') as f:
            W = pickle.load(f)
        for i in range(W.shape[0]):
            util.display_image(W.iloc[i, :])


class Experiments:

    def __init__(self):
        self.stam_obj = STAM()

    def check_training_accuracy(self, train=False):
        if train:
            self.stam_obj.run()
        test_data = self.stam_obj.make_df(self.stam_obj.test_img)
        test_label = self.stam_obj.test_lbl
        with open(SAVE_MODEL_PATH, 'rb') as f:
            W = pickle.load(f)
        W_lbl = [2, 1, 0, 7, 7, 3, 6, 5, 1, 3, 9, 4, 2, 6, 6]
        for T in range(test_data.shape[0]):
            # for each image in the test set, we will predict on them and check the accuracies of mse and simm
            self.stam_obj.predict(W, W_lbl, test_data.iloc[T, :], test_label[T])
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
            self.stam_obj.run()
        with open(SAVE_MODEL_PATH, 'rb') as f:
            W = pickle.load(f)
        valid_img = self.stam_obj.make_df(self.stam_obj.valid_img)
        labels = self.stam_obj.learn_labels(valid_img, self.stam_obj.valid_lbl, W)
        print labels

    def track_errors_during_training(self, num_of_clusters):
        data = self.stam_obj.make_df(self.stam_obj.train_img)
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

    def learn_w_with_less_clusters(self, num_clusters):
        self.stam_obj.run(num_clusters, display=True)

    def moving_average(self, values, window):
        weights = np.repeat(1.0, window) / window
        sma = np.convolve(values, weights, 'valid')
        return sma

    def run_experiments(self):
        #self.learn_w_with_less_clusters(10)
        #self.check_training_accuracy(train=False)
        #self.learn_labels_after_training(train=False)
        self.track_errors_during_training(num_of_clusters=10)


def main():
    expt = Experiments()
    expt.run_experiments()

if __name__=='__main__':
    main()







