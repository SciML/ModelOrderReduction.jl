module ModelOrderReduction
#========================Data Reduction=========================================#
include("Types.jl")
include("ErrorHandle.jl")

using LinearAlgebra
using TSVD
using RandomizedLinAlg

include("DataReduction/POD.jl")

export SVD, TSVD, RSVD
export POD, reduce!, matricize
#========================Model Reduction========================================#

#===============================================================================#
end
