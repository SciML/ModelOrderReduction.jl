module ModelOrderReduction

using DocStringExtensions
using Symbolics
using ModelingToolkit
using LinearAlgebra
using Setfield
using SparseArrays
using Bijections

include("utils.jl")

include("Types.jl")
include("ErrorHandle.jl")

include("DataReduction/POD.jl")
export SVD, TSVD, RSVD
export POD, reduce!

include("deim.jl")
export deim

include("polynomialize.jl")
export polynomialize

end
