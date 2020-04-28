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
# Recommendation: select the one with minimal complexity (k*l)
selected_model = top_results['best_n_models'][0]
