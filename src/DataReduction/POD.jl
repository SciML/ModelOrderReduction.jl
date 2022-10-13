using TSVD: tsvd
using RandomizedLinAlg: rsvd

function matricize(VoV::Vector{Vector{T}}) where {T}
    reduce(hcat, VoV)
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

mutable struct POD <: AbstractDRProblem
    # specified
    snapshots::Any
    min_renergy::Any
    min_nmodes::Int
    max_nmodes::Int
    # computed
    nmodes::Int
    rbasis::Any
    renergy::Any
    spectrum::Any
    # constructers
    function POD(snaps;
                 min_renergy = 1.0,
                 min_nmodes::Int = 1,
                 max_nmodes::Int = length(snaps[1]))
        nmodes = min_nmodes
        errorhandle(snaps, nmodes, min_renergy, min_nmodes, max_nmodes)
        new(snaps, min_renergy, min_nmodes, max_nmodes, nmodes, missing, 1.0, missing)
    end
    function POD(snaps, nmodes::Int)
        errorhandle(snaps, nmodes, 0.0, nmodes, nmodes)
        new(snaps, 0.0, nmodes, nmodes, nmodes, missing, 1.0, missing)
    end
end

function determine_truncation(s, min_nmodes, min_renergy, max_nmodes)
    nmodes = min_nmodes
    overall_energy = sum(s)
    energy = sum(s[1:nmodes]) / overall_energy
    while energy < min_renergy && nmodes < max_nmodes
        nmodes += 1
        energy += s[nmodes + 1] / overall_energy
    end
    return nmodes, energy
end

function reduce!(pod::POD, alg::SVD)
    u, s, v = _svd(pod.snapshots; alg.kwargs...)
    pod.nmodes, pod.renergy = determine_truncation(s, pod.min_nmodes, pod.max_nmodes,
                                                   pod.min_renergy)
    pod.rbasis = u[:, 1:(pod.nmodes)]
    pod.spectrum = s
    nothing
end

function reduce!(pod::POD, alg::TSVD)
    u, s, v = _tsvd(pod.snapshots, pod.nmodes; alg.kwargs...)
    n_max = min(size(u, 1), size(v, 1))
    pod.renergy = sum(s) / (sum(s) + (n_max - pod.nmodes) * s[end])
    pod.rbasis = u
    pod.spectrum = s
    nothing
end

function reduce!(pod::POD, alg::RSVD)
    u, s, v = _rsvd(pod.snapshots, pod.nmodes, alg.p)
    n_max = min(size(u, 1), size(v, 1))
    pod.renergy = sum(s) / (sum(s) + (n_max - pod.nmodes) * s[end])
    pod.rbasis = u
    pod.spectrum = s
    nothing
end

function Base.show(io::IO, pod::POD)
    print(io, "POD \n")
    print(io, "Reduction Order = ", pod.nmodes, "\n")
    print(io, "Snapshot size = (", size(pod.snapshots, 1), ",", size(pod.snapshots[1], 2),
          ")\n")
    print(io, "Relative Energy = ", pod.renergy, "\n")
end
