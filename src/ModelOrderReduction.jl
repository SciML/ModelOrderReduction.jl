module ModelOrderReduction

using DocStringExtensions

using Symbolics
using ModelingToolkit
using LinearAlgebra

using Setfield

include("utils.jl")

include("Types.jl")
include("ErrorHandle.jl")

using TSVD
using RandomizedLinAlg

include("DataReduction/POD.jl")
export SVD, TSVD, RSVD
export POD, reduce!

include("deim.jl")
export deim

end
