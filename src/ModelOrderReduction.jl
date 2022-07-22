module ModelOrderReduction
include("Types.jl")
include("ErrorHandle.jl")

using LinearAlgebra
using TSVD
using RandomizedLinAlg

include("DataReduction/POD.jl")

export SVD, TSVD, RSVD
export POD, reduce!, matricize

include("lift_learn.jl")

export polynomialization

end
