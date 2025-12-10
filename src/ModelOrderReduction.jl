module ModelOrderReduction

using DocStringExtensions

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

include("lift_and_learn.jl")
export lift_and_learn

end
