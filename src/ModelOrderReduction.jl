module ModelOrderReduction
#========================Data Reduction=========================================#
    include("Types.jl")
    include("ErrorHandle.jl")
    
    using LinearAlgebra
    using TSVD
    using RandomizedLinAlg

    include("DataReduction/POD.jl")
    include("DataReduction/DifussionMaps.jl")
    include("DataReduction/VAE.jl")

    export SVD, TSVD, RSVD
    export POD, reduce!, matricize
#========================Model Reduction========================================#

#===============================================================================#
end
