module ModelOrderReduction

using DocStringExtensions

using Symbolics
using ModelingToolkit
using LinearAlgebra

using Setfield

include("utils.jl")

include("Types.jl")
include("ErrorHandle.jl")

include("DataReduction/POD.jl")
export SVD, TSVD, RSVD
export POD, reduce!

include("deim.jl")
export deim

using PolyChaos
using Combinatorics: powerset, combinations
include("pce/pce.jl")
include("pce/pce_metadata.jl")
export PCE
include("pce/stochastic_galerkin.jl")
export stochastic_galerkin

end
