def iterate_cluster(matrix, modeltype, **kwargs):
    """
    Unifies clustering and evalluation into one step.
    Not recommended as computation can be time consuming and stepwise execution with saving in between is preferable.
    :param matrix: the data to be clustered
    :param modeltype:The clustering algorithm from Coclust to be used:
    "mod": CoclustMod(), "specmod": CoclustSpecMod(), "info": CoclustInfo()
    :param kwargs: kwargs to coclust_iterate()
    :return: list of results from functions coclust_iterate() and max_BIC_model()
    """
    models = coclust_iterate(matrix, modeltype, kwargs)
    bic_result = bic_models(models, matrix)
    print(bic_result[0])
    return bic_result, models