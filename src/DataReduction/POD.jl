using TSVD: tsvd
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

_tsvd(data, n::Int = 1; kwargs...) = tsvd(data, n; kwargs...)

function _rsvd(data::Vector{Vector{T}}, n::Int, p::Int) where {T}
    mat_data = matricize(data)
    return _rsvd(mat_data, n, p)
end

_rsvd(data, n::Int, p::Int) = rsvd(data, n, p)

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

function reduce!(pod::POD{S, T}, alg::TSVD)::Nothing where {S, T}
    u, s, v = _tsvd(pod.snapshots, pod.nmodes; alg.kwargs...)
    n_max = min(size(u, 1), size(v, 1))
    pod.renergy = T(sum(s) / (sum(s) + (n_max - pod.nmodes) * s[end]))
    pod.rbasis = Matrix{T}(u)
    pod.spectrum = Vector{T}(s)
    return nothing
end

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
