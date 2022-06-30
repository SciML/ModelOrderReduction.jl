# Leverage PCA from MultivariateStats.jl ?

function matricize(VoV::Vector{Vector{FT}}) where {FT}
    Matrix(reduce(hcat,VoV)')
end

mutable struct PCA{FT,IT} <: AbstractDRProblem
    snapshots::Union{Vector{Vector{FT}},Matrix{FT}}
    nmodes::IT
    rbasis::Vector{Vector{FT}}
    energy::FT

    function PCA(snaps::Vector{Vector{FT}},nmodes::IT) where {FT,IT}
        errorhandle(matricize(snaps),nmodes)
        new{eltype(snaps[1]),typeof(nmodes)}(snaps,nmodes)
    end

    function PCA(snaps::Matrix{FT},nmodes::IT) where {FT,IT}
        errorhandle(snaps,nmodes)
        new{eltype(snaps),typeof(nmodes)}(snaps,nmodes)
    end
end

function reduce!(pca::PCA{FT,IT}) where {FT,IT}
    op_matrix = pca.snapshots
    if typeof(pca.snapshots) == Vector{Vector{FT}}
        op_matrix = matricize(pca.snapshots)
    end
    u,s,v = svd(op_matrix)
    vr = v[:,1:pca.nmodes]
    sr = s[1:pca.nmodes]
    pca.energy = sum(sr)/sum(s)
    pca.rbasis = [vr[:,i] for i=1:pca.nmodes]
end

function Base.show(io::IO,pca::PCA)
    print(io,"PCA \n")
    print(io,"Reduction Order = ",pca.nmodes,"\n")
    print(io,"Snapshot size = ", length(pca.snapshots),",",length(pca.snapshots[1]),"\n")
    print(io,"Energy = ", pca.energy,"\n")
end
