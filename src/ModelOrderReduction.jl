module ModelOrderReduction

using DocStringExtensions: DocStringExtensions, FUNCTIONNAME, SIGNATURES, TYPEDSIGNATURES

using ModelingToolkit: ModelingToolkit, @variables, Differential, Equation, Num, ODESystem,
    SymbolicUtils, Symbolics, arguments, build_function, complete,
    expand, substitute, tearing_substitution
using LinearAlgebra: LinearAlgebra, /, \, mul!, qr, svd

using Setfield: Setfield, @set!

include("utils.jl")

include("Types.jl")
include("ErrorHandle.jl")

include("DataReduction/POD.jl")
export SVD, TSVD, RSVD
export POD, reduce!

include("deim.jl")
export deim

include("precompile.jl")

end
