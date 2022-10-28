"""
$(TYPEDEF)

# Fields
$(TYPEDFIELDS)
"""
struct PCE{P <: AbstractOrthoPoly}
    "Independent symbolic random variables ``X``."
    x::Vector{Num}
    "Univariate orthogonal polynomial basis for each ``X_i``."
    basis::Vector{P}
    "The array of multi-indices which could be a sparse grid."
    ind::Matrix{Int}
end
