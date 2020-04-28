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
    m, n = shape  # number of rows, cols
    count = int(m*n*proportion)

    sample = [1] * count
    zeros = [0] * (m * n - count)
    sample = sample + zeros
    random.shuffle(sample)
    sample = numpy.array(sample)
    sample.shape = shape
    return sample


def clustered_sample(shape, clusters, error=0.01):
    """
    Generate a binary matrix of given shape with a given number of diagonal clusters. The noise in the sample can be
    controlled with the error parameter.
    :param shape: shape of the matrix, tuple
    :param clusters: number of clusters in each variable
    :param error: noise in the distribution
    :return: list with two elements 0: the shuffled sample 1: the unshuffled sample
    """
    m, n = shape  # number of rows, cols

    size_x = int(n / clusters)
    size_y = int(m / clusters)
    new_n = size_x * clusters
    new_m = size_y * clusters
    sample = []
    for cluster in range(clusters):
        tmp = [0] * new_n
        indx = list(numpy.array(range(size_x)) + (cluster * size_x))
        for step, value in enumerate(tmp):
            if step in indx:
                tmp[step] = 1
        val = numpy.array(list(tmp) * size_y)
        noise = numpy.array(random.choices(population=[0, 1], weights=[1, error], k=len(val)))
        val = numpy.absolute(numpy.subtract(val, noise))
        sample.append(list(val))

    sample = numpy.array(sample)
    sample.shape = (new_m, new_n)
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


def no_0(value):
    """
    prevent absolute 0 values
    :param value:
    :return:
    """
    if value < 0.00000000001:
        value = 0.00000000001
    return value
