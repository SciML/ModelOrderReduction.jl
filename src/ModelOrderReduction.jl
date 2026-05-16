module ModelOrderReduction

using DocStringExtensions: DocStringExtensions, FUNCTIONNAME, SIGNATURES, TYPEDSIGNATURES

using ModelingToolkit: ModelingToolkit, @variables, @named, Differential, Equation, Num,
    ODESystem, System, SymbolicUtils, Symbolics, arguments, build_function, complete,
    equations, expand, iscall, operation, substitute, tearing_substitution, unknowns
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

include("PolynomializeQuadratizeReduce/PolynomializeQuadratizeReduceUtils.jl")
include("PolynomializeQuadratizeReduce/Polynomialization.jl")
include("PolynomializeQuadratizeReduce/Quadratization.jl")
include("PolynomializeQuadratizeReduce/GalerkinReduction.jl")

export polynomialize
export quadratize
export polynomialize_and_quadratize
export galerkin_project_system
export galerkin_project_system_affine
export compute_augmented_initial_pairs

include("precompile.jl")

end
