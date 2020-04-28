from datetime import datetime
import numpy
from coclust.coclustering import CoclustSpecMod, CoclustMod, CoclustInfo
from optimal_cocluster.bic import bic_cocluster
import warnings
from optimal_cocluster.helperfunctions import progress


def coclust_models(matrix,
                   modeltype,
                   n_col_clusters=None,
                   n_row_clusters=None,
                   n_clusters=None,
                   savemodel=False,
                   namestr="",
                   path="",
                   **kwargs):
    """
    Generates a range of co-clustering solutions using package Coclust.
    :param matrix: dataset to be clustered
    :param modeltype: The clustering algorithm from Coclust to be used:
    "mod": CoclustMod(), "specmod": CoclustSpecMod(), "info": CoclustInfo()
    :param n_col_clusters: number of column clusters if modeltype == "info"
    :param n_row_clusters: number of row clusters if modeltype == "info"
    :param n_clusters: number of clusters to be used if modeltype != "info"
    :param savemodel: bool, TRUE saves the resulting list of models using pickle under path
    :param namestr: optional name, used in filename when pickling
    :param path: optional path to a directory, used when pickling
    :param kwargs: kwargs to clustering functions from Cuclust
    :return: list of coclustering models
    """
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
            progress(c + 1, n_clusters)
        else:
            for r in range(n_row_clusters):
                model = cluster(n_row_clusters=c + 1, n_col_clusters=r + 1, **kwargs)
                model.fit(matrix)
                models.append(model)
                step = r + 1 + c * n_row_clusters
                progress(step, n_clusters * n_row_clusters)
    print("... completed.")

    if savemodel:
        try:
            import pickle
            with open(path + namestr + modeltype + 'Models' + str(n_col_clusters) + "-" + str(n_row_clusters),
                      'wb') as f:
                pickle.dump(models, f)
        except ModuleNotFoundError:
            pass

    return models


def bic_models(models,
               matrix):
    """
    evaluates a list of Coclust models using Bayesian Information Criterion and identifies the model with max BIC
    :param models: list of coclustering models (generated by coclust_iterate())
    :param matrix: data to be clustered
    :param
    :return: complete bic list(corresponds to the model list)
    """
    evaluation = {"Models": [],
                  "BIC": [],
                  "Approximate Distribution": [],
                  "Cocluster Distribution": []}
    now = datetime.now()
    print(now.strftime("%d%m%H:%M"), "Evaluating ", len(models), "models ...")
    for step, model in enumerate(models):
        result = bic_cocluster(colpartition=model.column_labels_, rowpartition=model.row_labels_, matrix=matrix)
        evaluation["Models"].append(model)
        evaluation["BIC"].append(result["BIC"])
        evaluation["Approximate Distribution"].append(result["Approximate Distribution"])
        evaluation["Cocluster Distribution"].append(result["Cocluster Distribution"])
        progress(step + 1, len(models))
    print("... completed.")
    return evaluation


def process_results(evaluation,
                    n_top=5):
    """
    Find the n models with the highest BIC
    :param evaluation: results from bic_models()
    :param n_top: n_top: number of top solutions to output
    :return: dict {"best_model": the best model,
        "best_bic": bic value of the best model,
        "best_n_models": n best models,
        "best_n_bic": bic of n best models,
        "best_n_approx_dist": approximate distribution of n best models,
        "best_co_dist": cocluster distribution of n best models,
    """
    top = (-numpy.array(evaluation["BIC"])).argsort()[:n_top]
    top_models = numpy.array(evaluation["Models"])[top]
    top_bic = numpy.array(evaluation["BIC"])[top]
    top_approx_dist = numpy.array(evaluation["Approximate Distribution"])[top]
    top_co_dist = numpy.array(evaluation["Cocluster Distribution"])[top]
    return {
        "best_model": top_models[0],
        "best_bic": top_bic[0],
        "best_n_models": top_models,
        "best_n_bic": top_bic,
        "best_n_approx_dist": top_approx_dist,
        "best_co_dist": top_co_dist}
