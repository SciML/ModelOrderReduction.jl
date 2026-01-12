abstract type AbstractReductionProblem end
abstract type AbstractMORProblem <: AbstractReductionProblem end
abstract type AbstractDRProblem <: AbstractReductionProblem end

abstract type AbstractSVD end

struct SVD{K <: NamedTuple} <: AbstractSVD
    kwargs::K
    function SVD(; kwargs...)
        kw = NamedTuple(kwargs)
        return new{typeof(kw)}(kw)
    end
end

struct TSVD{K <: NamedTuple} <: AbstractSVD
    kwargs::K
    function TSVD(; kwargs...)
        kw = NamedTuple(kwargs)
        return new{typeof(kw)}(kw)
    end
end

struct RSVD <: AbstractSVD
    p::Int
    function RSVD(p::Int = 0)
        return new(p)
    end
end
