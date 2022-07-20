module ModelOrderReduction
#========================Data Reduction=========================================#
include("Types.jl")
include("ErrorHandle.jl")

using LinearAlgebra
using TSVD
using RandomizedLinAlg

import TSVD.tsvd
import RandomizedLinAlg.rsvd
import LinearAlgebra.svd

include("DataReduction/POD.jl")

export SVD, TSVD, RSVD
export POD, reduce!, matricize
#========================Model Reduction========================================#

#===============================================================================#
end
