abstract type AbstractReductionProblem end
abstract type AbstractMORProblem <: AbstractReductionProblem end
abstract type AbstractDRProblem <: AbstractReductionProblem end

abstract type AbstractSVD end

"""
    SVD(; kwargs...)

Dense singular value decomposition backend for projection basis construction.
"""
struct SVD{K <: NamedTuple} <: AbstractSVD
    kwargs::K
    function SVD(; kwargs...)
        kw = NamedTuple(kwargs)
        return new{typeof(kw)}(kw)
    end
end

"""
    TSVD(; kwargs...)

Truncated singular value decomposition backend for projection basis construction.
"""
struct TSVD{K <: NamedTuple} <: AbstractSVD
    kwargs::K
    function TSVD(; kwargs...)
        kw = NamedTuple(kwargs)
        return new{typeof(kw)}(kw)
    end
end

"""
    RSVD([p])

Randomized singular value decomposition backend with oversampling parameter `p`.
"""
struct RSVD <: AbstractSVD
    p::Int
    function RSVD(p::Int = 0)
        return new(p)
    end
end
