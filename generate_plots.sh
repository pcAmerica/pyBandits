#!/bin/bash
# Generate test data
python -m unittest bandits

# Generate graphs using R
Rscript r/epsilon_greedy/plot_standard.R
Rscript r/epsilon_greedy/plot_annealing.R
Rscript r/ucb/plot_ucb1.R
Rscript r/comparisons/plot_simple_comparisons.R
Rscript r/comparisons/plot_comparisons.R

#Rscript r/softmax/plot_standard.R
#Rscript r/softmax/plot_annealing.R
#Rscript r/exp3/plot_exp3.R