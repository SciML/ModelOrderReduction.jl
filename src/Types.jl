abstract type AbstractReductionProblem end
abstract type AbstractMORProblem <: AbstractReductionProblem end
abstract type AbstractDRProblem <: AbstractReductionProblem end

abstract type AbstractSVD end

"""
    SVD(; kwargs...)

Dense singular value decomposition backend for [`reduce!`](@ref).

# Keyword Arguments
- `kwargs...`: keyword arguments forwarded to `LinearAlgebra.svd`.

# Fields
- `kwargs`: the named tuple of keyword arguments forwarded to the decomposition.

# Examples
```jldoctest
julia> using ModelOrderReduction

julia> SVD() isa SVD
true
```
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

Truncated singular value decomposition backend for [`reduce!`](@ref). Use this backend
when only the requested reduced modes should be computed.

# Keyword Arguments
- `kwargs...`: keyword arguments forwarded to `TSVD.tsvd`.

# Fields
- `kwargs`: the named tuple of keyword arguments forwarded to the decomposition.

# Examples
```jldoctest
julia> using ModelOrderReduction

julia> TSVD() isa TSVD
true
```
"""
struct TSVD{K <: NamedTuple} <: AbstractSVD
    kwargs::K
    function TSVD(; kwargs...)
        kw = NamedTuple(kwargs)
        return new{typeof(kw)}(kw)
    end
end

"""
    RSVD([p = 0])

Randomized singular value decomposition backend for [`reduce!`](@ref).

# Arguments
- `p::Integer = 0`: number of oversampling vectors used by `RandomizedLinAlg.rsvd`.

# Fields
- `p::Int`: the oversampling parameter.

# Examples
```jldoctest
julia> using ModelOrderReduction

julia> RSVD(2).p
2
```
"""
struct RSVD <: AbstractSVD
    p::Int
    function RSVD(p::Int = 0)
        return new(p)
    end
end
