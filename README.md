OptimalCoclust - evaluating cocluster quality
==============================================

This package provides simple tools to evaulate coclustering solutions.
The package is made to work with the CoClust-Package (https://pypi.org/project/coclust/ and https://github.com/franrole/cclust_package) but should be usable with other methods of two-way clustering.

Evaluation of cluster quality:
-----------------------------
Here, co-cluster solutions are evaluated by determining the Bayesian Information Criterion (BIC). 
In the application of co-clustering the common method for determining BIC is not directly applicable 
and has to be adjusted for the two dimensions of the model.

This implementation employs an adoption of BIC for two dimensional models presented by:

    Cai, R., Lu, L., & Hanjalic, A. (2008). Co-clustering for auditory scene categorization. IEEE Transactions on multimedia, 10(4), 596-606.
(https://www.researchgate.net/publication/3424749_Co-clustering_for_Auditory_Scene_Categorization)

The resulting value describes the amount of mutual information retained between the original data and the approximation through clustering 
in relation to model complexity. When comparing models, higher values indicate that a models approximation retains more information
and is therefore preferable.

Key functions
-------------

General workflow:
----------------
![example output](imgpath) 

Example:
--------

```bash
"""
Recommended Workflow for identifying the optimal combination of clusters
"""
from optimal_cocluster import iterate
from optimal_cocluster.helperfunctions import random_sample
from optimal_cocluster.iterate import identify_top_bic

binary_sample = random_sample((200, 150), 0.3)

# set the maximum of clusters to be evaluated
c = 20

# run and store the models for all [c,c] combinations
example_models = iterate.coclust_models(matrix=binary_sample,
                                        n_row_clusters=c,
                                        n_col_clusters=c,
                                        modeltype="info",
                                        namestr="example",
                                        savemodel=True,
                                        path="")

# evaluate models
evaluation = iterate.bic_models(models=example_models, matrix=binary_sample)

# extract results for top 5 best models
top_results = identify_top_bic(evaluation=evaluation, n_top=5)

# further extract the model from the top models
# Recommendation: select the with minimal complexity (k*l)
selected_model = top_results['best_n_models'][0]
```
