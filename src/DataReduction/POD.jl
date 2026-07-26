import TSVD as TruncatedSVD
using RandomizedLinAlg: rsvd

function matricize(VoV::Vector{Vector{T}})::Matrix{T} where {T}
    return reduce(hcat, VoV)
end

function _svd(data::Vector{Vector{T}}; kwargs...) where {T}
    mat_data = matricize(data)
    return _svd(mat_data; kwargs...)
end

_svd(data; kwargs...) = svd(data; kwargs...)

function _tsvd(data::Vector{Vector{T}}, n::Int = 1; kwargs...) where {T}
    mat_data = matricize(data)
    return _tsvd(mat_data, n; kwargs...)
end

_tsvd(data, n::Int = 1; kwargs...) = TruncatedSVD.tsvd(data, n; kwargs...)

function _rsvd(data::Vector{Vector{T}}, n::Int, p::Int) where {T}
    mat_data = matricize(data)
    return _rsvd(mat_data, n, p)
end

_rsvd(data, n::Int, p::Int) = rsvd(data, n, p)

"""
    POD(snapshots; min_renergy = 1.0, min_nmodes = 1, max_nmodes = length(snapshots[1]))
    POD(snapshots, nmodes)

Proper orthogonal decomposition reduction problem built from state snapshots. Call
[`reduce!`](@ref) with an SVD backend to compute its basis and spectrum.

# Arguments
- `snapshots`: a state-by-snapshot matrix or a vector of state vectors.
- `nmodes::Int`: the fixed number of retained modes. This positional form disables
  energy-based truncation.

# Keyword Arguments
- `min_renergy = 1.0`: minimum captured relative spectral energy when selecting modes.
- `min_nmodes::Int = 1`: lower bound on the number of retained modes.
- `max_nmodes::Int = length(snapshots[1])`: upper bound on the number of retained modes.

# Fields
- `snapshots`: the input snapshot data.
- `min_renergy`, `min_nmodes`, `max_nmodes`: the truncation policy.
- `nmodes`: the selected number of retained modes.
- `rbasis`: the reduced basis after [`reduce!`](@ref), or `missing` before reduction.
- `renergy`: the captured relative spectral energy.
- `spectrum`: the singular-value spectrum after [`reduce!`](@ref), or `missing` before
  reduction.

# Examples
```jldoctest
julia> using ModelOrderReduction

julia> pod = POD([3.0 0.0; 0.0 1.0], 1);

julia> pod.nmodes
1
```
"""
mutable struct POD{S, T <: AbstractFloat} <: AbstractDRProblem
    # specified
    snapshots::S
    min_renergy::T
    min_nmodes::Int
    max_nmodes::Int
    # computed
    nmodes::Int
    rbasis::Union{Missing, Matrix{T}}
    renergy::T
    spectrum::Union{Missing, Vector{T}}
    # constructors
    function POD(
            snaps::S;
            min_renergy::T = 1.0,
            min_nmodes::Int = 1,
            max_nmodes::Int = length(snaps[1])
        ) where {S <: AbstractMatrix{T}} where {T <: AbstractFloat}
        nmodes = min_nmodes
        errorhandle(snaps, nmodes, min_renergy, min_nmodes, max_nmodes)
        return new{S, T}(snaps, min_renergy, min_nmodes, max_nmodes, nmodes, missing, one(T), missing)
    end
    function POD(
            snaps::S;
            min_renergy::T = 1.0,
            min_nmodes::Int = 1,
            max_nmodes::Int = length(snaps[1])
        ) where {T <: AbstractFloat, S <: AbstractVector{<:AbstractVector{T}}}
        nmodes = min_nmodes
        errorhandle(snaps, nmodes, min_renergy, min_nmodes, max_nmodes)
        return new{S, T}(snaps, min_renergy, min_nmodes, max_nmodes, nmodes, missing, one(T), missing)
    end
    function POD(snaps::S, nmodes::Int) where {S <: AbstractMatrix{T}} where {T <: AbstractFloat}
        errorhandle(snaps, nmodes, zero(T), nmodes, nmodes)
        return new{S, T}(snaps, zero(T), nmodes, nmodes, nmodes, missing, one(T), missing)
    end
    function POD(snaps::S, nmodes::Int) where {T <: AbstractFloat, S <: AbstractVector{<:AbstractVector{T}}}
        errorhandle(snaps, nmodes, zero(T), nmodes, nmodes)
        return new{S, T}(snaps, zero(T), nmodes, nmodes, nmodes, missing, one(T), missing)
    end
end

function determine_truncation(
        s::AbstractVector{T}, min_nmodes::Int, max_nmodes::Int, min_renergy::T
    )::Tuple{Int, T} where {T <: AbstractFloat}
    nmodes = min_nmodes
    overall_energy = sum(s)
    energy = sum(s[1:nmodes]) / overall_energy
    while energy < min_renergy && nmodes < max_nmodes
        nmodes += 1
        energy += s[nmodes + 1] / overall_energy
    end
    return nmodes, energy
end

"""
    reduce!(pod::POD, alg::SVD) -> nothing

Compute the reduced basis and full singular-value spectrum for `pod` using dense SVD.

# Arguments
- `pod::POD`: reduction problem to update in place.
- `alg::SVD`: dense singular value decomposition backend.

# Examples
```jldoctest
julia> using ModelOrderReduction

julia> pod = POD([3.0 0.0; 0.0 1.0], 1);

julia> reduce!(pod, SVD()); size(pod.rbasis)
(2, 1)
```
"""
function reduce!(pod::POD{S, T}, alg::SVD)::Nothing where {S, T}
    u, s, v = _svd(pod.snapshots; alg.kwargs...)
    pod.nmodes,
        pod.renergy = determine_truncation(
        s, pod.min_nmodes, pod.max_nmodes,
        pod.min_renergy
    )
    pod.rbasis = Matrix{T}(u[:, 1:(pod.nmodes)])
    pod.spectrum = Vector{T}(s)
    return nothing
end

"""
    reduce!(pod::POD, alg::TSVD) -> nothing

Compute the reduced basis and truncated singular-value spectrum for `pod` using a
truncated SVD.

# Arguments
- `pod::POD`: reduction problem to update in place.
- `alg::TSVD`: truncated singular value decomposition backend.
"""
function reduce!(pod::POD{S, T}, alg::TSVD)::Nothing where {S, T}
    u, s, v = _tsvd(pod.snapshots, pod.nmodes; alg.kwargs...)
    n_max = min(size(u, 1), size(v, 1))
    pod.renergy = T(sum(s) / (sum(s) + (n_max - pod.nmodes) * s[end]))
    pod.rbasis = Matrix{T}(u)
    pod.spectrum = Vector{T}(s)
    return nothing
end

"""
    reduce!(pod::POD, alg::RSVD) -> nothing

Compute the reduced basis and approximate singular-value spectrum for `pod` using a
randomized SVD.

# Arguments
- `pod::POD`: reduction problem to update in place.
- `alg::RSVD`: randomized singular value decomposition backend.
"""
function reduce!(pod::POD{S, T}, alg::RSVD)::Nothing where {S, T}
    u, s, v = _rsvd(pod.snapshots, pod.nmodes, alg.p)
    n_max = min(size(u, 1), size(v, 1))
    pod.renergy = T(sum(s) / (sum(s) + (n_max - pod.nmodes) * s[end]))
    pod.rbasis = Matrix{T}(u)
    pod.spectrum = Vector{T}(s)
    return nothing
end

function Base.show(io::IO, pod::POD)::Nothing
    print(io, "POD \n")
    print(io, "Reduction Order = ", pod.nmodes, "\n")
    print(
        io, "Snapshot size = (", size(pod.snapshots, 1), ",", size(pod.snapshots[1], 2),
        ")\n"
    )
    print(io, "Relative Energy = ", pod.renergy, "\n")
    return nothing
end
