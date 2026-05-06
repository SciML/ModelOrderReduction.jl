module ModelOrderReduction

using DocStringExtensions: DocStringExtensions, FUNCTIONNAME, SIGNATURES, TYPEDSIGNATURES

using ModelingToolkit: ModelingToolkit, @variables, @named, Differential, Equation, Num,
    ODESystem, System, SymbolicUtils, Symbolics, arguments, build_function, complete,
    expand, substitute, equations, unknowns, tearing_substitution
using OrdinaryDiffEq: ODEProblem, Tsit5, solve
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
include("PolynomializeQuadratizeReduce/PolynomializeQuadratizeReduce.jl")

export polynomialize
export quadratize
export galerkin_project_system
export galerkin_project_system_affine
export polynomialize_quadratize_reduce

include("precompile.jl")

end
