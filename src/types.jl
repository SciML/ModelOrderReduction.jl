abstract type AbstractReductionProblem end
abstract type AbstractMORProblem <: AbstractReductionProblem end
abstract type AbstractDRProblem <: AbstractReductionProblem end

abstract type AbstractSVD end
struct SVD <: AbstractSVD end
struct TSVD <: AbstractSVD end
struct RSVD <: AbstractSVD end

export SVD, TSVD, RSVD
