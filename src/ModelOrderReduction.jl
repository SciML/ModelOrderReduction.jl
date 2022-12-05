module ModelOrderReduction

using Combinatorics: powerset, combinations
using DocStringExtensions
using LinearAlgebra
using ModelingToolkit
using PolyChaos
using RandomizedLinAlg: rsvd
using Setfield
using SparseArrays
using Symbolics
using TSVD: tsvd

include("utils.jl")

include("Types.jl")
include("ErrorHandle.jl")
include("DataReduction/POD.jl")
export POD, reduce!, matricize

include("deim.jl")
export deim

include("pce/pce.jl")
export PCE
include("pce/pce_metadata.jl")
include("pce/stochastic_galerkin.jl")
export stochastic_galerkin

end
