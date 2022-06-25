# Leverage PCA from MultivariateStats.jl ?

function matricize(VoV::Vector{Vector{FT}}) where {FT}
    Matrix(reduce(hcat,VOV)')
end

mutable struct PCA{FT,IT} <: AbstractDRProblem
    snapshots::Vector{Vector{FT}}
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

function Base.reduce!(pca::PCA{FT,IT}) where {FT,IT}
    op_matrix = snapshots
    if typeof(snapshots) == Vector{Vector{FT}}
        op_matrix = matricize(snapshots)
    end
    u,s,v = svd(op_matrix)
    vr = v[:,1:pca.nmodes]
    sr = s[1:pca.nmodes]
    pca.energy = sum(sr)/sum(s)
    pca.rbasis = [vr[:,i] for i=1:pca.nmodes]
end

function Base.show(io::IO,pca::PCA)

end
