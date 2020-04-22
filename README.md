Optimal-Coclust - evaluating cocluster quality
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
example code
```
