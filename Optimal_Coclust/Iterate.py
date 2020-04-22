import sys
from datetime import datetime
import numpy
from coclust.coclustering import CoclustSpecMod, CoclustMod, CoclustInfo
from Optimal_Coclust.BIC import bic_cocluster
import warnings


def coclust_iterate(matrix,
                    modeltype,
                    n_col_clusters=None,
                    n_row_clusters=None,
                    n_clusters=None,
                    savemodel=False,
                    namestr="",
                    path="",
                    **kwargs):
    warnings.filterwarnings("ignore", category=FutureWarning)
    cluster = None
    n_iterations = n_clusters
    if modeltype == "mod":
        cluster = CoclustMod
    elif modeltype == "specmod":
        cluster = CoclustSpecMod
    elif modeltype == "info":
        cluster = CoclustInfo
        n_clusters = n_col_clusters
        n_iterations = n_col_clusters * n_row_clusters

    models = []
    now = datetime.now()
    print(now.strftime("%d%m%H:%M"), "Clustering ", n_iterations, "iterations ...")
    for c in range(n_clusters):
        if modeltype == "mod" or modeltype == "specmod":
            model = cluster(n_clusters=c + 1, **kwargs)
            model.fit(matrix)
            models.append(model)
            progress(c+1,n_clusters)
        else:
            for r in range(n_row_clusters):
                model = cluster(n_row_clusters=c + 1, n_col_clusters=r + 1, **kwargs)
                model.fit(matrix)
                models.append(model)
                step = r + 1 + c * n_row_clusters
                progress(step, n_clusters*n_row_clusters)
    print("... completed.")

    if savemodel:
        try:
            import pickle
            with open(path + namestr + modeltype + 'Models' + str(n_col_clusters)+ "-"+ str(n_row_clusters), 'wb') as f:
                pickle.dump(models, f)
        except ModuleNotFoundError:
            pass

    return models


def max_BIC_model(models, matrix):
    BIC_collect = []
    now = datetime.now()
    print(now.strftime("%d%m%H:%M"), "Evaluating ", len(models), "models ...")
    for step,model in enumerate(models):
        result = bic_cocluster(colpartition=model.column_labels_, rowpartition=model.row_labels_, matrix=matrix)
        BIC_collect.append(result[0])
        progress(step+1, len(models))
    print("... completed.")
    maxBIC = numpy.where(BIC_collect == numpy.nanmax(BIC_collect))[0][0]
    return models[int(maxBIC)], BIC_collect


def iterate_cluster(matrix, modeltype,**kwargs):
    models = coclust_iterate(matrix, modeltype, kwargs)
    bic_result = max_BIC_model(models, matrix)
    print(bic_result[0])
    return bic_result, models


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
