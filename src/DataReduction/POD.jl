function matricize(VoV::AbstractVector{T}) where T <: AbstractVector
    Matrix(reduce(hcat, VoV))
end

function svd(data::AbstractVector{T}) where T <: AbstractVector 
    mat_data = matricize(data)
    return svd(mat_data)
end

function svd(data::Vector{T}) where T <: AbstractVector
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
    max_renergy 
    min_nmodes::Int
    max_nmodes::Int 
    function POD(snaps::AbstractVector{T}, nmodes::Int; max_renergy = 1.0,
                 max_nmodes::Int = nmodes) where {T <: AbstractVector}
        errorhandle(snaps, nmodes, max_renergy, nmodes, max_nmodes)
        new{typeof(snaps)}(snaps, nmodes, nothing, 0.0, max_renergy, nmodes, max_nmodes)
    end

    function POD(snaps::AbstractMatrix, nmodes::Int = 1; max_renergy = 1.0,
                 max_nmodes::Int = nmodes) 
        errorhandle(snaps, nmodes, max_renergy, nmodes, max_nmodes)
        new{typeof(snaps)}(snaps, nmodes, nothing, 0.0, max_renergy, nmodes, max_nmodes)
    end
end

function determine_truncation(s, min_nmodes, max_renergy, max_nmodes)
    nmodes = min_nmodes
    overall_energy = sum(s)
    energy = sum(s[1:nmodes])/overall_energy
    while energy < max_renergy && nmodes <= max_nmodes
        nmodes += 1
        energy += s[nmodes+1]/overall_energy
    end
    return nmodes, energy
end

function reduce!(pod::POD, ::SVD)
    u, s, v = svd(pod.snapshots)
    pod.nmodes, pod.renergy = determine_truncation(s, pod.min_nmodes, pod.max_nmodes, pod.max_renergy)
    pod.rbasis = u[:, 1:(pod.nmodes)]
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
