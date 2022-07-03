module ModelOrderReduction
#========================Data Reduction=========================================#
    include("Types.jl")
    include("ErrorHandle.jl")

    using LinearAlgebra
    using TSVD
    using RandomizedLinAlg

    include("DataReduction/POD.jl")
    include("DataReduction/DiffusionMaps.jl")
    include("DataReduction/VAE.jl")

    export SVD, TSVD, RSVD
    export POD, reduce!, matricize
#========================Model Reduction========================================#
    include("ModelReduction/LiftAndLearn.jl")

#===============================================================================#
end
