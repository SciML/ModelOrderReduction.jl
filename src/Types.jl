"""
    AbstractReductionProblem

Internal common supertype for model-reduction problem types.

This type is not exported and is not a supported extension point. User code should
construct one of the documented concrete problem types instead.
"""
abstract type AbstractReductionProblem end

"""
    AbstractMORProblem <: AbstractReductionProblem

Internal marker type for model-order-reduction problems.

This type is not exported and is not a supported extension point. Its hierarchy may
change without a public API guarantee.
"""
abstract type AbstractMORProblem <: AbstractReductionProblem end

"""
    AbstractDRProblem <: AbstractReductionProblem

Internal marker type for data-reduction problems such as [`POD`](@ref).

This type is not exported and is not a supported extension point. Use the documented
[`POD`](@ref) constructors and [`reduce!`](@ref) methods instead.
"""
abstract type AbstractDRProblem <: AbstractReductionProblem end

"""
    AbstractSVD

Internal supertype for the built-in singular-value-decomposition backends.

This type is not exported and is not a supported extension point. The supported
backends are [`SVD`](@ref), [`TSVD`](@ref), and [`RSVD`](@ref); custom subtypes and
additional `reduce!` dispatches are not part of the public API.
"""
abstract type AbstractSVD end

"""
    SVD(; kwargs...) -> SVD

Dense singular value decomposition backend for [`reduce!`](@ref).

# Keywords
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
    TSVD(; kwargs...) -> TSVD

Truncated singular value decomposition backend for [`reduce!`](@ref). Use this backend
when only the requested reduced modes should be computed.

# Keywords
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
    RSVD([p::Int = 0]) -> RSVD

Randomized singular value decomposition backend for [`reduce!`](@ref).

# Arguments
- `p::Int = 0`: number of oversampling vectors used by `RandomizedLinAlg.rsvd`.

# Fields
- `p::Int`: the oversampling parameter.

# Throws
- `MethodError`: if `p` is not an `Int`.

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
