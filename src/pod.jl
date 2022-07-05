using LinearAlgebra

"""
    pod(snapshots::AbstractMatrix, dim::Integer)

Compute the Proper Orthogonal Decomposition (POD) for a model with a snapshot matrix and a
POD dimension. Return the POD basis and singular values.

The results are sorted in descending order of singular values.

The POD dimension should be smaller than the number of snapshots.

# Arguments
- `snapshots::AbstractMatrix`: a matrix with snapshots in the columns.
- `dim::Integer`: the POD dimension.

# Examples
```jldoctest
julia> n_eq = 10; # number of equations

julia> n_snapshot = 6; # number of snapshots

julia> snapshots = rand(n_eq, n_snapshot);

julia> dim = 3; # POD dimension

julia> pod_basis, singular_vals = pod(snapshots, dim);

julia> size(pod_basis)
(10, 3)

julia> size(singular_vals)
(3,)
```
"""
function pod(snapshots::AbstractMatrix, dim::Integer)
    eigen_vecs, singuler_vals = svd(snapshots)
    if size(snapshots, 2) < dim
        @warn "The POD dimension is larger than the number of snapshots"
        return eigen_vecs, singuler_vals
    else
        return (@view eigen_vecs[:, 1:dim]), (@view singuler_vals[1:dim])
    end
end

export pod
