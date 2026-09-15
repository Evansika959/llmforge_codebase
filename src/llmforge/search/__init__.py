"""Search spaces, NSGA-II, and the co-search dispatcher.

    individual      the architecture record and its analytic cost estimates
    elastic_space   the space of slices of one trained supernet
    hetero_space    the heterogeneous from-scratch space
    nsga2           NSGA-II population, constrained domination, crowding, checkpoints
    pareto          non-dominated filtering and exact hypervolume
    cosearch        command-line dispatcher that joins a software and a hardware evaluator
"""
