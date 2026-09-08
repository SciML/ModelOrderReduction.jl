module ModelOrderReduction

using DocStringExtensions: DocStringExtensions, FUNCTIONNAME, TYPEDEF, TYPEDFIELDS,
    TYPEDSIGNATURES

using ModelingToolkit: ModelingToolkit, @parameters, @variables, Differential, Equation, Num,
    System, SymbolicUtils, Symbolics, complete, mtkcompile, substitute
using SciMLBase: SciMLBase
using SymbolicIndexingInterface: SymbolicIndexingInterface
using LinearAlgebra: LinearAlgebra, /, \, mul!, qr, svd
using SparseArrays: SparseArrays, SparseMatrixCSC, findnz, sparse

include("Types.jl")
include("ErrorHandle.jl")

include("DataReduction/POD.jl")
export SVD, TSVD, RSVD
export POD, reduce!

include("full_order.jl")
include("deim.jl")
export deim

include("pod.jl")
export pod

include("precompile.jl")

end
