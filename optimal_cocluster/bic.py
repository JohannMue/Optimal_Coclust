import numpy as np
from optimal_cocluster.helperfunctions import no_0


def bic_cocluster(matrix, colpartition=None, rowpartition=None, appromatrix=None):
    """
    Calculate Bayesian information criterion for a co-cluster solution
    Should be maximized!

    based on Kullback-Leibler divergence between the original distribution and the approximation based on clustering
    weighted for model complexity
    Adapted from:
    Cai, R., Lu, L., & Hanjalic, A. (2008). Co-clustering for Auditory Scene Categorization. IEEE Transactions on
    Multimedia, 10(4), 596–606. https://doi.org/10.1109/TMM.2008.921739
    :param matrix: Contingency table or joint distribution
    :param rowpartition: vector of integer row cluster labels
    :param colpartition: vector of integer  column cluster labels
    :param appromatrix: Contingency table or joint distribution for clustered solution (optional) if not provided, this
    table is calculated from row and col partitions

    :return:
    """
    rowpartition = standardize_partition(rowpartition)
    colpartition = standardize_partition(colpartition)
    matrix = np.array(matrix)
    # calculate probabilities from data
    # normalize matrix from 0 to 1
    matrix = 1.0 * matrix / np.sum(matrix, axis=None, keepdims=True)
    co_dist = co_joint_dist(matrix=matrix,
                            rowpartition=rowpartition,
                            colpartition=colpartition)
    if appromatrix is None:

        approx_dist = approx_joint_dist(matrix=matrix,
                                        rowpartition=rowpartition,
                                        colpartition=colpartition,
                                        co_dist=co_dist)
    else:
        approx_dist = np.array(appromatrix)

    # normalize matrix from 0 to 1
    approx_dist = 1.0 * approx_dist / np.sum(approx_dist, axis=None, keepdims=True)

    # dimensionality
    m, n = matrix.shape  # number of rows, cols
    lmda = m * n  # tuning parameter

    # calculate mutual information of contingency tables
    # before clustering
    i_ori = _matrix_information(matrix)
    # after clustering
    i_star = _matrix_information(approx_dist)

    mi_relation = _mi__relation(i1=i_ori, i2=i_star)  # data likelihood
    complexity = _complexity(matrix=matrix, colpartition=colpartition, rowpartition=rowpartition)
    bic = lmda * mi_relation - complexity
    return {"BIC": bic,
            "Approximate Distribution": approx_dist,
            "Cocluster Distribution": co_dist}


def _complexity(matrix, colpartition, rowpartition):
    k = len(np.unique(rowpartition))  # number of rowclusters
    l = len(np.unique(colpartition))  # number of colclusters
    m, n = matrix.shape  # number of rows

    complexity = (((n * k) / 2) * np.log(m) + ((m * l) / 2) * np.log(n))
    return complexity


def _mi__relation(i1, i2):
    information_relation = no_0(i2 / i1)
    mi_relation = np.log(information_relation)  # data likelihood
    return mi_relation


def _matrix_information(matrix):
    """
    determines mutual information of a joint distribution
    For more information see: Dhillon, I. S., Mallela, S., & Modha, D. S. (2003, August). Information-theoretic
    co-clustering. In Proceedings of the ninth ACM SIGKDD international conference on Knowledge discovery and
    data mining (pp. 89-98).
    :param matrix: joint distribution or approximation
    :return: mutual information
    """
    matrix = 1.0 * matrix / np.sum(matrix, axis=None, keepdims=True)
    information = 0
    for s in range(matrix.shape[0]):
        for f in range(matrix.shape[1]):
            p_sf = no_0(matrix[s, f])
            p_s = no_0(np.sum(matrix[s]))
            p_f = no_0(np.sum(matrix[:, f]))
            information = information + p_sf * np.log2(p_sf / (p_s * p_f))
    return information


def approx_joint_dist(matrix,
                      rowpartition,
                      co_dist,
                      colpartition):
    """
    calculate an approximation of joint distribution based on a distribution and a corresponding co-clustering solution
    expressed in row and column labels
    :param matrix: the initial distribution, usually empirical
    :param co_dist: joint distribution of coclusters, from  co_joint_dist()
    :param rowpartition: list of row cluster labels
    :param colpartition: list of column cluster labels
    :return: 2d arrayjoint distribution approximated from original distribution and clustering
    """
    # normalize matrix from 0 to 1
    matrix = 1.0 * matrix / np.sum(matrix, axis=None, keepdims=True)

    rowpartition = standardize_partition(rowpartition)
    colpartition = standardize_partition(colpartition)
    # use partitioning to determine approximate probabilities
    # Marginal probabilities
    p_x = np.sum(matrix, axis=0)
    p_y = np.sum(matrix, axis=1)

    p_cxy = co_dist
    # Marginal probabilities of joint distribution
    p_cx = np.sum(p_cxy, axis=0)
    p_cy = np.sum(p_cxy, axis=1)

    # approximate joint distribution
    p_xy = matrix * 0.0
    for l_indx, l in enumerate(colpartition):
        for k_indx, k in enumerate(rowpartition):
            p_kl = p_cxy[k, l]
            a = p_x[l_indx] / no_0(p_cx[l])
            b = p_y[k_indx] / no_0(p_cy[k])
            temp = p_kl * a * b
            p_xy[k_indx, l_indx] = temp
    return p_xy


def co_joint_dist(matrix,
                  rowpartition,
                  colpartition):
    """
    calculates cocluster joint distribution from data and partitioning
    :param matrix: the initial distribution, usually empirical
    :param rowpartition: list of row cluster labels
    :param colpartition: list of column cluster labels
    :return: 2d array cocluster joint distribution
    """
    rowpartition = standardize_partition(rowpartition)
    colpartition = standardize_partition(colpartition)
    p_cxy = np.array([0.0] * (len(np.unique(rowpartition)) * len(np.unique(colpartition))))
    p_cxy.shape = (len(np.unique(rowpartition)), len(np.unique(colpartition)))
    for l in set(colpartition):
        for k in set(rowpartition):
            cols = list(np.where(np.array(colpartition) == l)[0])
            rows = list(np.where(np.array(rowpartition) == k)[0])
            p_kl = np.sum(matrix[np.ix_(rows, cols)], axis=None)
            p_cxy[k, l] = p_kl
    return p_cxy


def standardize_partition(partition):
    """
    standardize a cluster partitioning into an ascending integer list
    :param partition: list of partition labels, str, int or otherwise
    :return: standardized integer labels
    """
    std_partition = [np.where(i == np.unique(np.array(partition)))[0][0] for i in np.array(partition)]
    return std_partition
