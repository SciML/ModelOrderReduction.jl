module ModelOrderReduction

using DocStringExtensions
using Symbolics
using ModelingToolkit
using LinearAlgebra
using Setfield
using SparseArrays
using PolyChaos
using Combinatorics: powerset, combinations

include("utils.jl")

include("Types.jl")
include("ErrorHandle.jl")

include("DataReduction/POD.jl")
export SVD, TSVD, RSVD
export POD, reduce!

include("deim.jl")
export deim

include("pce/pce.jl")
export PCE
include("pce/pce_metadata.jl")
include("pce/stochastic_galerkin.jl")
export stochastic_galerkin

end
