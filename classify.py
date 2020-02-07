import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf

#  size of training and testing set
TRAIN_SIZE = 60000
TEST_SIZE = 10000

#  number of images to plot
PLOT_SIZE = 9

def main():

    (trainingImages, trainingLabels), (testingImages, testingLabels) = tf.keras.datasets.mnist.load_data()

    # select size of dataset to train and test on
    # trainingImages, trainingLabels = trainingImages[:TRAIN_SIZE], trainingLabels[:TRAIN_SIZE]
    # testingImages, testingLabels = testingImages[:TEST_SIZE], testingLabels[:TEST_SIZE]

    # flatten the images into one dimension
    trainingImages = np.reshape(trainingImages, (TRAIN_SIZE, 28 * 28)).astype(dtype=np.float64)
    testingImages = np.reshape(testingImages, (TEST_SIZE, 28 * 28)).astype(dtype=np.float64)

    # transpose the matrices so they are # pixels x # images
    trainingImages = np.swapaxes(trainingImages, 0, 1)
    testingImages = np.swapaxes(testingImages, 0, 1)
    trainingImages = trainingImages.astype(np.float64)

    meanColVectors, eigVectors = hw1FindEigendigits(trainingImages)

    pixels = trainingImages[:,:PLOT_SIZE].astype(dtype='uint8').reshape((PLOT_SIZE, 28, 28))


    ######### Plot images #########
    # plt.style.use('seaborn-muted')
    #
    # fig, axes = plt.subplots(3, 3, figsize=(5, 5), sharex=True, sharey=True)
    #
    # for i in range(PLOT_SIZE):
    #     # axes (subplot) objects are stored in 2d array, accessed with axes[row,col]
    #     subplot_row = i // 3
    #     subplot_col = i % 3
    #     ax = axes[subplot_row, subplot_col]
    #
    #     # plot image on subplot
    #     img = pixels[i, :]
    #     ax.imshow(img, cmap='gray_r')
    #
    #     ax.set_title('Digit Label: {}'.format(trainingLabels[i]))
    #     ax.set_xbound([0, 28])
    #
    # plt.tight_layout()
    # plt.show()
    # plt.clf()
    f = plt.figure()
    plt.imshow(pixels[0, :], cmap='gray_r')
    # plt.show()

    # reconstruct digits from eigenspace


def hw1FindEigendigits(A:np.array) -> (np.array, np.array):
    """
        Returns mean column vector of the training images
        and eigenvectors of covariance matrix
    """
    m, n = A.shape

    # number of eigenvectors to return
    T = 100

    meanColVectors = np.mean(A, axis=1).reshape((m,1))

    # normalize by mean of col vectors
    A -= meanColVectors

    # find eigenvectors and eigenvalues from covariance matrix
    covMatrix = np.matmul(A, np.transpose(A))
    eigValues, eigVectors =  np.linalg.eig(covMatrix)

    # sort the eigenvectors in descending order by eigenvalues
    indexArray = np.flip(np.argsort(eigValues))
    eigVectors = eigVectors[indexArray]

    # select top T eigenvectors
    topT = eigVectors[:,:T]

    # normalize eigenvectors
    eigVectors = np.linalg.norm(topT, 1, axis=1)

    return meanColVectors, eigVectors

main()