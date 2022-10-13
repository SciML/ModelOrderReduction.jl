module ModelOrderReduction
#========================Data Reduction=========================================#
using DocStringExtensions
using Symbolics
using ModelingToolkit
using LinearAlgebra

include("Types.jl")
include("ErrorHandle.jl")
include("DataReduction/POD.jl")

export SVD, TSVD, RSVD
export POD, reduce!, matricize

#========================Model Reduction========================================#
# Discrete Empirical Interpolation
using Setfield
include("DEIM/utils.jl")
include("DEIM/deim.jl")
export deim

# Polynomial Chaos Expansion
include("PCE/PCE.jl")

end
