# Leverage PCA from MultivariateStats.jl ?

function matricize(VoV::Vector{Vector{FT}}) where {FT}
    Matrix(reduce(hcat,VoV)')
end

mutable struct POD{FT,IT} <: AbstractDRProblem
    snapshots::Union{Vector{Vector{FT}},Matrix{FT}}
    nmodes::IT
    rbasis::Vector{Vector{FT}}
    energy::FT

    function POD(snaps::Vector{Vector{FT}},nmodes::IT) where {FT,IT}
        errorhandle(matricize(snaps),nmodes)
        new{eltype(snaps[1]),typeof(nmodes)}(snaps,nmodes)
    end

    function POD(snaps::Matrix{FT},nmodes::IT) where {FT,IT}
        errorhandle(snaps,nmodes)
        new{eltype(snaps),typeof(nmodes)}(snaps,nmodes)
    end
end

function reduce!(pod::POD{FT,IT},::SVD) where {FT,IT}
    op_matrix = pod.snapshots
    if typeof(pod.snapshots) == Vector{Vector{FT}}
        op_matrix = matricize(pod.snapshots)
    end
    u,s,v = svd(op_matrix)
    vr = v[:,1:pod.nmodes]
    sr = s[1:pod.nmodes]
    pod.energy = sum(sr)/sum(s)
    pod.rbasis = [vr[:,i] for i=1:pod.nmodes]
    nothing
end

function reduce!(pod::POD{FT,IT},::TSVD) where {FT,IT}
    op_matrix = pod.snapshots
    if typeof(pod.snapshots) == Vector{Vector{FT}}
        op_matrix = matricize(pod.snapshots)
    end
    u,s,v = tsvd(op_matrix,pod.nmodes)
    pod.energy = NaN
    pod.rbasis = [v[:,i] for i=1:pod.nmodes]
    nothing
end

function reduce!(pod::POD{FT,IT},::RSVD) where {FT,IT}
    op_matrix = pod.snapshots
    if typeof(pod.snapshots) == Vector{Vector{FT}}
        op_matrix = matricize(pod.snapshots)
    end
    u,s,v = rsvd(op_matrix,pod.nmodes)
    pod.energy = NaN
    pod.rbasis = [v[:,i] for i=1:pod.nmodes]
    nothing
end

function Base.show(io::IO,pod::POD)
    print(io,"POD \n")
    print(io,"Reduction Order = ",pod.nmodes,"\n")
    print(io,"Snapshot size = (", length(pod.snapshots),",",length(pod.snapshots[1]),")\n")
    print(io,"Energy = ", pod.energy,"\n")
end
