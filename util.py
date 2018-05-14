#@author-Joseph_Aribido
import os
from mnist import MNIST
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim as ssim

PATH = '/home/ribex/Desktop/Dev/neural_algo/dataset'

def load():

    mndata = MNIST(PATH)
    train_img, train_lbl = mndata.load_training()
    test_img, test_lbl = mndata.load_testing()
    return np.asarray(train_img), np.asarray(train_lbl), np.asarray(test_img), np.asarray(test_lbl)

def display_image(image, rows=28, cols=28):
    plt.imshow(np.asarray(image).reshape(rows, cols), interpolation='nearest')
    plt.show()

def display_digits(image, rows=28, cols=28):
    pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    values = np.asarray(image).reshape(rows, cols)
    print (pd.DataFrame(values))

def mse_two_images(image1, image2, rows=28, cols=28):
    image1 = np.asarray(image1).astype(np.double).reshape(rows, cols)
    image2 = np.asarray(image2).astype(np.double).reshape(rows, cols)
    mse_err = 1/(cols*rows)*np.sum(np.square(image1 - image2))
    return mse_err

def ssim_two_images(image1, image2, rows=28, cols=28):
    image1 = np.asarray(image1).astype(np.double).reshape(rows, cols)
    image2 = np.asarray(image2).astype(np.double).reshape(rows, cols)
    return ssim(image1, image2)