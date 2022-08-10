"""
$(TYPEDSIGNATURES)

Compute the DEIM interpolation indices for the given projection basis.

The orthonormal `basis` should not be a sparse matrix.
"""
function deim_interpolation_indices(basis::AbstractMatrix)::Vector{Int}
    dim = size(basis, 2)
    indices = Vector{Int}(undef, dim)
    indices[1] = argmax(abs(x) for x in @view basis[:, 1])
    for l in 2:dim
        U = @view basis[:, 1:(l - 1)]
        P = @view indices[1:(l - 1)]
        PᵀU = @view U[P, :]
        uₗ = @view basis[:, l]
        Pᵀuₗ = @view uₗ[P, :]
        c = PᵀU \ Pᵀuₗ
        r = vec(uₗ - U * c)
        indices[l] = argmax(abs(x) for x in r)
    end
    return indices
end

# 1. compute DEIM interpolation indices
# 2. compute DEMI projector
# 3. transform the nonlinear functions F
function deim_project(basis::AbstractMatrix, pod_dict::Dict, F)
    indices = deim_interpolation_indices(basis) # DEIM interpolation indices
    # the DEIM projector (not DEIM basis) satisfies
    # F(original_vars) ≈ projector * F(pod_basis * reduced_vars)[indices]
    projector = ((@view basis[indices, :])' \ basis')'
    temp = substitute.(F[indices], (pod_dict,))
    projector * temp # DEIM approximation for nonlinear func F
end

"""
$(TYPEDSIGNATURES)

Reduce an ODESystem using the Discrete Empirical Interpolation Method (DEIM).

The DEIM relies on the Proper Orthogonal Decomposition (POD). The given `pod_basis` should
be an orthonormal basis matrix with POD modes in the columns.

The LHS of `sys` are all assumed to be 1st order derivatives. Use
`ModelingToolkit.ode_order_lowering` to transform higher order ODEs before applying DEIM.

`sys` is assumed to have no internal systems. End users are encouraged to call
`ModelingToolkit.structural_simplify` beforehand, which calls
`ModelingToolkit.expand_connections` internally.

`deim_basis` is default to be the same as `pod_basis`, as the POD basis is normally a
suitable choice for the DEIM index selection algorithm. Users can also provide their own
DEIM basis and/or choose a lower dimension for DEIM than POD.
"""
function deim(sys::ODESystem, pod_basis::AbstractMatrix;
              deim_basis::AbstractMatrix = pod_basis,
              deim_dim::Integer = size(pod_basis, 2),
              name::Symbol = Symbol(nameof(sys), "_deim"))::ODESystem
    iv = ModelingToolkit.get_iv(sys) # the single independent variable
    D = Differential(iv)
    dvs = ModelingToolkit.get_states(sys) # dependent variables
    deqs, eqs = get_deqs(sys) # differential and non-differential equations

    rhs = Symbolics.rhss(deqs)
    F = polynomial_coeffs(rhs, dvs)[2] # non-polynomial nonlinear part
    polynomial = rhs - F # polynomial terms
    pod_dim = size(pod_basis, 2) # the dimension of POD basis

    y_pod = Symbolics.variables(:y_pod, 1:pod_dim; T = Symbolics.FnType)
    y_pod = map(y -> y(iv), y_pod) # new variables from POD reduction

    pod_eqs = Symbolics.scalarize(dvs .~ pod_basis * y_pod)
    pod_dict = Dict(eq.lhs => eq.rhs for eq in pod_eqs) # original vars to reduced vars
    reduced_polynomial = substitute.(polynomial, (pod_dict,))
    inv_dict = Dict(collect(y_pod) .=> pod_basis' * dvs) # reduced vars to orignial vars

    U = @view deim_basis[:, 1:deim_dim] # DEIM projection basis
    deim_nonlinear = deim_project(U, pod_dict, F)
    deqs = D.(y_pod) .~ pod_basis' * (reduced_polynomial + deim_nonlinear)

    new_eqs = [Symbolics.scalarize(deqs); eqs]
    new_oberved = [observed(sys); pod_eqs]
    new_defaults = merge(ModelingToolkit.defaults(sys), inv_dict)

    ODESystem(new_eqs, iv, y_pod, parameters(sys);
              observed = new_oberved, name = name,
              defaults = new_defaults,
              continuous_events = ModelingToolkit.continuous_events(sys), checks = false)
end

export deim
