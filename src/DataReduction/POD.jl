function matricize(VoV::AbstractVector{T}) where T <: AbstractVector
    Matrix(reduce(hcat, VoV))
end

function svd(data::AbstractVector{T}) where T <: AbstractVector 
    mat_data = matricize(data)
    return svd(mat_data)
end

function tsvd(data::AbstractVector{T}, n::Int) where T <: AbstractVector 
    mat_data = matricize(data)
    return tsvd(mat_data, n)
end

function rsvd(data::AbstractVector{T}, n::Int, p::Int) where T <: AbstractVector 
    mat_data = matricize(data)
    return rsvd(mat_data, n, p)
end

mutable struct POD{snapType} <: AbstractDRProblem
    snapshots::snapType
    nmodes::Int
    rbasis
    renergy

    function POD(snaps::AbstractVector{T}, nmodes::Int) where {T <: AbstractVector}
        errorhandle(snaps, nmodes)
        new{typeof(snaps)}(snaps, nmodes, nothing, 0.0)
    end

    function POD(snaps::AbstractMatrix, nmodes::Int)
        errorhandle(snaps, nmodes)
        new{typeof(snaps)}(snaps, nmodes, nothing, 0.0)
    end
end

function reduce!(pod::POD, ::SVD)
    u, s, v = svd(pod.snapshots)
    pod.rbasis = u[:, 1:(pod.nmodes)]
    sr = s[1:(pod.nmodes)]
    pod.renergy = sum(sr) / sum(s)
    nothing
end

function reduce!(pod::POD, ::TSVD)
    u, s, v = tsvd(pod.snapshots, pod.nmodes)
    n_max = min(size(u,1), size(v,1))
    pod.renergy = sum(s) / (sum(s) + (n_max - pod.nmodes) * s[end])
    pod.rbasis = u
    nothing
end

function reduce!(pod::POD, rsvd_alg::RSVD)
    u, s, v = rsvd(pod.snapshots, pod.nmodes, rsvd_alg.p)
    n_max = min(size(u,1), size(v,1))
    pod.renergy = sum(s) / (sum(s) + (n_max - pod.nmodes) * s[end])
    pod.rbasis = u
    nothing
end

function Base.show(io::IO, pod::POD)
    print(io, "POD \n")
    print(io, "Reduction Order = ", pod.nmodes, "\n")
    print(io, "Snapshot size = (", size(pod.snapshots, 1), ",", size(pod.snapshots[1], 2),
          ")\n")
    print(io, "Relative Energy = ", pod.renergy, "\n")
end
