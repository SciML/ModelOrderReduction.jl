"""
Abstract type for quadratization algorithms.
"""
abstract type QuadratizationAlgorithm end

"""
    QBee <: QuadratizationAlgorithm

QBee (Quadratization via monomial introduction) algorithm from Bychkov & Pogudin (2021).

This algorithm finds an optimal monomial quadratization by introducing new variables
that represent monomials in the original state variables. It minimizes the number of
new variables needed to transform the system to quadratic form.

# Reference
Bychkov, A., & Pogudin, G. (2021). Optimal monomial quadratization for ODE systems.
https://arxiv.org/abs/2103.08013
"""
struct QBee <: QuadratizationAlgorithm end
