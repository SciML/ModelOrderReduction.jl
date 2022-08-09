abstract type AbstractReductionProblem end
abstract type AbstractMORProblem <: AbstractReductionProblem end
abstract type AbstractDRProblem <: AbstractReductionProblem end

abstract type AbstractSVD end

struct SVD <: AbstractSVD
    kwargs::Any
    function SVD(; kwargs...)
        new(kwargs)
    end
end

struct TSVD <: AbstractSVD
    kwargs::Any
    function TSVD(; kwargs...)
        new(kwargs)
    end
end

struct RSVD <: AbstractSVD
    p::Int
    function RSVD(p::Int = 0)
        new(p)
    end
end
