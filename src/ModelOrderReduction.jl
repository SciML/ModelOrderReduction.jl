module ModelOrderReduction

using DocStringExtensions

using Symbolics
using ModelingToolkit

include("utils.jl")

include("Types.jl")
include("ErrorHandle.jl")

using LinearAlgebra
using TSVD
using RandomizedLinAlg

include("DataReduction/POD.jl")

export SVD, TSVD, RSVD
export POD, reduce!, matricize

include("deim.jl")

end
