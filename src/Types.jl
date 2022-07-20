abstract type AbstractReductionProblem end
abstract type AbstractMORProblem <: AbstractReductionProblem end
abstract type AbstractDRProblem <: AbstractReductionProblem end

abstract type AbstractSVD end
struct SVD <: AbstractSVD end
struct TSVD <: AbstractSVD end
struct RSVD <: AbstractSVD 
    p::Int

    function RSVD(p::Int=0)
        new(p)
    end
end
