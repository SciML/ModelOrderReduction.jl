using LinearAlgebra
using TSVD
using RandomizedLinAlg

function errorhandle(data::Matrix{FT}, modes::IT) where {FT, IT}
    @assert size(data, 1)>1 "State vector is expected to be vector valued."
    s = size(data, 2)
    @assert (modes > 0)&(modes < s) "Number of modes should be in {1,2,...,$(s-1)}."
end

function matricize(VoV::Vector{Vector{FT}}) where {FT}
    Matrix(reduce(hcat, VoV))
end

mutable struct POD{FT} <: AbstractDRProblem
    snapshots::Union{Vector{Vector{FT}}, Matrix{FT}}
    nmodes::Int
    rbasis::Matrix{FT}
    renergy::FT

    function POD(snaps::Vector{Vector{FT}}, nmodes::Int) where {FT}
        errorhandle(matricize(snaps), nmodes)
        new{eltype(snaps[1])}(snaps, nmodes, Array{FT, 2}(undef, size(snaps, 1), nmodes),
                              FT(0))
    end

    function POD(snaps::Matrix{FT}, nmodes::Int) where {FT}
        errorhandle(snaps, nmodes)
        new{eltype(snaps)}(snaps, nmodes, Array{FT, 2}(undef, size(snaps, 1), nmodes),
                           FT(0))
    end

    function POD(snaps::Adjoint{FT, Matrix{FT}}, nmodes::Int) where {FT}
        POD(Matrix(snaps), nmodes, Array{FT, 2}(undef, size(snaps, 1), nmodes), FT(0))
    end
end

function reduce!(pod::POD{FT}, ::SVD) where {FT}
    op_matrix = pod.snapshots
    if typeof(pod.snapshots) == Vector{Vector{FT}}
        op_matrix = matricize(pod.snapshots)
    end
    u, s, v = svd(op_matrix)
    pod.rbasis .= u[:, 1:(pod.nmodes)]
    sr = s[1:(pod.nmodes)]
    pod.renergy = sum(s) / (sum(s) + (size(op_matrix, 1) - pod.nmodes) * s[end])
    nothing
end

function reduce!(pod::POD{FT}, ::TSVD) where {FT}
    op_matrix = pod.snapshots
    if typeof(pod.snapshots) == Vector{Vector{FT}}
        op_matrix = matricize(pod.snapshots)
    end
    u, s, v = tsvd(op_matrix, pod.nmodes)
    pod.renergy = sum(s) / (sum(s) + (size(op_matrix, 1) - pod.nmodes) * s[end])
    pod.rbasis .= u
    nothing
end

function reduce!(pod::POD{FT}, ::RSVD) where {FT}
    op_matrix = pod.snapshots
    if typeof(pod.snapshots) == Vector{Vector{FT}}
        op_matrix = matricize(pod.snapshots)
    end
    u, s, v = rsvd(op_matrix, pod.nmodes)
    pod.renergy = sum(s) / (sum(s) + (size(op_matrix, 1) - pod.nmodes) * s[end])
    pod.rbasis .= u
    nothing
end

function Base.show(io::IO, pod::POD)
    print(io, "POD \n")
    print(io, "Reduction Order = ", pod.nmodes, "\n")
    print(io, "Snapshot size = (", size(pod.snapshots, 1), ",", size(pod.snapshots[1], 2),
          ")\n")
    print(io, "Relative Energy = ", pod.renergy, "\n")
end

export POD, reduce!, matricize
