import sys
import numpy
import random


def random_sample(shape, proportion):
    """
    generate a random binary matrix of a provided shape and a given proportion of 1
    :param shape: shape of matrix, tuple
    :param proportion: proportion [0;1] of 1 in the sample
    :return: binary matrix of given shape
    """
    x = shape[0]
    y = shape[1]
    count = int(x*y*proportion)

    sample = [1] * count
    zeros = [0] * (x * y - count)
    sample = sample + zeros
    random.shuffle(sample)
    sample = numpy.array(sample)
    sample.shape = shape
    return sample


def clustered_sample(shape, clusters, error=0.01):
    """
    Generate a binary matrix of given shape with a given number of diagonal clusters. The noise in the sample can be controlled with the
    error parameter.
    :param shape: shape of the matrix, tuple
    :param clusters: number of clusters in each variable
    :param error: noise in the distribution
    :return: list with two elements 0: the shuffled sample 1: the unshuffled sample
    """
    x = shape[0]
    y = shape[1]
    size_x = int(x / clusters)
    size_y = int(y / clusters)
    x = size_x * clusters
    y = size_y * clusters
    sample = []
    for cluster in range(clusters):
        tmp = [0] * x
        indx = list(numpy.array(range(size_x)) + (cluster * size_x))
        for step, value in enumerate(tmp):
            if step in indx:
                tmp[step] = 1
        val = numpy.array(list(tmp) * size_y)
        noise = numpy.array(random.choices(population=[0, 1], weights=[1, error], k=len(val)))
        val = numpy.absolute(numpy.subtract(val, noise))
        sample.append(list(val))

    sample = numpy.array(sample)
    sample.shape = (y, x)
    sample_shuffle = sample
    numpy.random.shuffle(sample_shuffle)
    numpy.random.shuffle(numpy.transpose(sample_shuffle))
    return sample, sample_shuffle


def progress(step, total):
    """
    a lightweight console progress bar
    adopted from https://stackoverflow.com/a/3002100
    :param step: the current step
    :param total: max steps
    :return:
    """
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('=' * int(step/total*20), int(step/total*100)))
    sys.stdout.flush()