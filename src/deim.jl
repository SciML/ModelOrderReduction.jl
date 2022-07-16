using Symbolics, ModelingToolkit

"""
$(TYPEDSIGNATURES)

Compute the DEIM interpolation indices for the given projection basis.

`basis` should not be a sparse matrix.
"""
function deim_interpolation_indices(basis::AbstractMatrix)::Vector{Int}
    dim = size(basis, 2)
    indices = Vector{Int}(undef, dim)
    indices[1] = argmax(abs.(@view basis[:, 1]))
    for l in 2:dim
        U = @view basis[:, 1:(l - 1)]
        P = @view indices[1:(l - 1)]
        PᵀU = @view U[P, :]
        uₗ = @view basis[:, l]
        Pᵀuₗ = @view uₗ[P, :]
        c = PᵀU \ Pᵀuₗ
        r = vec(uₗ - U * c)
        indices[l] = argmax(abs.(r))
    end
    return indices
end

"""
$(TYPEDSIGNATURES)

Reduce an ODESystem using the Discrete Empirical Interpolation Method (DEIM).

The DEIM relies on the Proper Orthogonal Decomposition (POD).  The given `pod_basis` is a
basis matrix with POD modes in the columns.

The LHS of `sys` are all assumed to be 1st order derivatives. Use
`ModelingToolkit.ode_order_lowering` to transform higher order ODEs before applying DEIM.

The DEIM basis `deim_basis` is default to be the same as `pod_basis`, as the POD basis is
normally a suitable choice for the DEIM index selection algorithm.

```jldoctest
julia> const N = 10; # number of variables

julia> using ModelingToolkit; @variables t u[1:N](t);

julia> D = Differential(t);

julia> using SparseArrays; A = sprand(N, N, 0.3);

julia> f(x) = sin(x);

julia> @named sys = ODESystem(D.(u) .~ A * u + f.(u));

julia> const n_snapshot = 2N; # random number for POD snapshots for illustration

julia> const pod_dim = 5; # random number for POD dimension

julia> using LinearAlgebra

julia> pod_basis = @view svd(rand(N, n_snapshot)).U[:, 1:pod_dim]; # random orthormal basis

julia> deim_sys = deim(sys, pod_basis); # DEIM reduced system

julia> length(states(deim_sys)) # number of variables
5

julia> length(equations(deim_sys)) # number of equations
5
```
"""
function deim(sys::ODESystem, pod_basis::AbstractMatrix;
              deim_basis::AbstractMatrix = pod_basis,
              deim_dim::Integer = size(pod_basis, 2),
              name::Symbol = Symbol(nameof(sys), "_deim"))::ODESystem
    iv = ModelingToolkit.get_iv(sys) # the single independent variable
    dvs = states(sys) # dependent variables
    rhs = Symbolics.rhss(equations(sys))
    F = polynomial_coeffs(rhs, dvs)[2] # non-polynomial nonlinear part
    polynomial = rhs - F # polynomial terms
    U = @view deim_basis[:, 1:deim_dim] # DEIM projection basis
    indices = deim_interpolation_indices(U) # DEIM interpolation indices
    D = Differential(iv)
    pod_dim = size(pod_basis, 2) # the dimension of POD basis
    @variables y_pod[1:pod_dim](iv) # new variables from POD reduction
    pod_eqs = Symbolics.scalarize(dvs .~ pod_basis * y_pod)
    pod_dict = Dict(eq.lhs => eq.rhs for eq in pod_eqs) # original vars to reduced vars
    reduced_polynomial = substitute.(polynomial, (pod_dict,))
    # the DEIM projector (not DEIM basis) satisfies
    # F(original_vars) ≈ projector * F(pod_basis * reduced_vars)[indices]
    projector = ((@view U[indices, :])' \ U')'
    deim_nonlinear = substitute.(F[indices], (pod_dict,))
    deim_nonlinear = projector * deim_nonlinear # DEIM approximation for nonlinear func F
    inv_dict = Dict(collect(y_pod) .=> pod_basis' * dvs) # reduced vars to orignial vars
    ODESystem(D.(y_pod) .~ pod_basis' * (reduced_polynomial + deim_nonlinear);
              observed = [observed(sys); pod_eqs], name = name,
              defaults = merge(ModelingToolkit.defaults(sys), inv_dict))
end

export deim
