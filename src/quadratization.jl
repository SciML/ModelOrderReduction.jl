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

"""
    quadratize(sys::System, alg::QuadratizationAlgorithm = QBee(); 
               name = Symbol(nameof(sys), :_quadratized), kwargs...)

Transform a polynomial ODE system into quadratic form using the specified algorithm.

# Arguments
- `sys::System`: Input polynomial ODE system with first-order derivatives on LHS
- `alg::QuadratizationAlgorithm`: Algorithm to use for quadratization (default: `QBee()`)
- `name`: Name for the output system (default: name of input system)
- `kwargs...`: Algorithm-specific keyword arguments

# Returns
- `System`: Quadratic ODE system with introduced auxiliary variables
"""
function quadratize(
        sys::System, alg::QuadratizationAlgorithm = QBee();
        name = Symbol(nameof(sys), :_quadratized), kwargs...
    )::System
    return quadratize(sys, alg, name; kwargs...)
end
function quadratize(sys::System, alg::QBee, name; kwargs...)::System
    return quadratize_qbee(sys, name; kwargs...)
end

function quadratize_qbee(sys::System, name; kwargs...)
    # TODO
    return sys
end
