import numpy as np


def bic_cocluster(matrix, colpartition=None, rowpartition=None, appromatrix=None):
    """
    Calculate Bayesian information criterion for a co-cluster solution
    Should be maximized!

    based on Kullback-Leibler divergence between the original distribution and the approximation based on clustering
    weighted for model complexity
    Adapted from:
    Cai, R., Lu, L., & Hanjalic, A. (2008). Co-clustering for Auditory Scene Categorization. IEEE Transactions on Multimedia, 10(4), 596–606. https://doi.org/10.1109/TMM.2008.921739
    :param matrix: Contingency table or joint distribution
    :param rowpartition: vector of integer row cluster labels
    :param colpartition: vector of integer  column cluster labels
    :param appromatrix: Contingency table or joint distribution for clustered solution (optional) if not provided, this
    table is calculated from row and col partitions

    :return: list containing BIC, lmda, mi_relation, complexity
    BIC:  Bayesian information criterion
    lmda: tuning parameter for weighting the data likelihood
    mi_relation: data likelihood based on mutual information
    complexity: data complexity
    BIC is calculated from lmda * mi_relation - complexity
    Values are provided separately to allow use of different lamda values without repeating the clustering
    """
    rowpartition = standardize_partition(rowpartition)
    colpartition = standardize_partition(colpartition)
    matrix = np.array(matrix)
    # calculate probabilities from data
    # normalize matrix from 0 to 1
    matrix = 1.0 * matrix / np.sum(matrix, axis=None, keepdims=True)

    if appromatrix is None:
        approx_matrix = approximation_from_partition(matrix=matrix, rowpartition=rowpartition, colpartition=colpartition)[0]
    else:
        approx_matrix = np.array(appromatrix)

    # normalize matrix from 0 to 1
    approx_matrix = 1.0 * approx_matrix / np.sum(approx_matrix, axis=None, keepdims=True)

    # calculate mutual information
    # dimensionality
    m = matrix.shape[0]  # number of rows
    n = matrix.shape[1]  # number of cols

    k = len(np.unique(rowpartition))  # number of rowclusters
    l = len(np.unique(colpartition))  # number of colclusters
    # calculate mutual information of contingency tables
    # before clustering
    I = _matrix_information(matrix)
    # after clustering
    I_star = _matrix_information(approx_matrix)

    lmda = m * n  # weight
    information_relation = I_star / I
    # log(0) prevention
    if information_relation <= 0.00000000001:
        information_relation = 0.00000000001
    mi_relation = np.log(information_relation)  # data likelihood
    complexity = (((n * k) / 2) * np.log(m) + ((m * l)/ 2) * np.log(n))
    BIC = lmda * mi_relation - complexity
    return BIC, lmda, mi_relation, complexity


def _matrix_information(matrix):
    I = 0
    for s in range(matrix.shape[0]):
        for f in range(matrix.shape[1]):
            p_sf = matrix[s, f]
            if p_sf == 0:
                p_sf = 0.00000000001
            p_s = np.sum(matrix[s])
            if p_s == 0:
                p_s = 0.00000000001
            p_f = np.sum(matrix[:, f])
            if p_f == 0:
                p_f = 0.00000000001
            I = I + p_sf * np.log2(p_sf / (p_s * p_f))
    return I

def approximation_from_partition(matrix,rowpartition, colpartition):
    rowpartition = standardize_partition(rowpartition)
    colpartition = standardize_partition(colpartition)
    # use partitioning to determine approximate probabilities
    # edgesums
    colsums = np.sum(matrix, axis=(0))
    rowsums = np.sum(matrix, axis=(1))

    # edge cluster probability
    colsums_clustprop = []
    for colclust in colpartition:
        index = list(np.where(np.array(colpartition) == colclust)[0])
        colsums_clustprop.append(np.sum(colsums[index]))

    rowsums_clustprop = []
    for rowclust in rowpartition:
        index = list(np.where(np.array(rowpartition) == rowclust)[0])
        rowsums_clustprop.append(np.sum(rowsums[index]))

    # calculate approximate probabilities (joint distribution) from
    # a = probability from joint distribution
    # b = column probability relative to colcluster probability
    # c = row probability relative to rowcluster probability
    approx_matrix = matrix * 0
    joint_dist = np.array([0] * (len(np.unique(rowpartition))*len(np.unique(colpartition))))
    joint_dist.shape = (len(np.unique(rowpartition)),len(np.unique(colpartition)))

    for colindx, colval in enumerate(colpartition):
        for rowindx, rowval in enumerate(rowpartition):
            cols = list(np.where(np.array(colpartition) == colval)[0])
            rows = list(np.where(np.array(rowpartition) == rowval)[0])
            a = np.sum(matrix[np.ix_(rows, cols)], axis=None)
            joint_dist[rowval,colval] = a
            b = colsums[colindx] / colsums_clustprop[colindx]
            c = rowsums[rowindx] / rowsums_clustprop[rowindx]
            temp = a * b * c
            approx_matrix[rowindx, colindx] = temp
    return approx_matrix,joint_dist


def standardize_partition(partition):
    """
    standardize a cluster partitioning into an ascending integer list
    :param partition: list of partition labels, str, int or otherwise
    :return: standardized integer labels
    """
    p = [np.where(i == np.unique(np.array(partition)))[0][0] for i in np.array(partition)]
    return p
