# screening_ncvx_penalty

This repository contains the code for the paper "Screening Rules for Lasso with Non-Convex Sparse Regularizers" published at ICML 2019, Long Beach.

The current version of the paper is available at https://arxiv.org/pdf/1902.06125.pdf
For abstract, bibliography (.bib), you can look at http://proceedings.mlr.press/v97/rakotomamonjy19a.html


this repository is still in construction.

# installation

download the source code from the git repository

# known dependencies

python (>= 3.6) , scipy (1.1.0), numpy (1.15.4). 

# structure of the repository


- screening_lasso.py contains the code for solving lasso, weighted lasso and proximal weighted lasso using coordinatewise descent

- noncvx_lasso.py contains codes for solving non-convex lasso using coordinate wise descent, with screening, and with screening and screening propagation

- compare_algorithms.py allows to reproduce experiments using toy data. note that the case (n=500, d=5000) may take several hours especially for bcd

- generate_figure.py allows to generate figures based on the saved results from "compare_algorithms". 


for reproducing figure 1 (left) in the paper, run compare_algorithms.py as is and then run generate_figure.py




