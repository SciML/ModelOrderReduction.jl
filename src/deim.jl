"""
$(TYPEDSIGNATURES)

Compute the DEIM interpolation indices for the given projection basis.

The orthonormal `basis` should not be a sparse matrix.
"""
function deim_interpolation_indices(basis::AbstractMatrix)::Vector{Int}
    dim = size(basis, 2)
    indices = Vector{Int}(undef, dim)
    @inbounds @views begin
        r = abs.(basis[:, 1])
        indices[1] = argmax(r)
        for l in 2:dim
            U = basis[:, 1:(l - 1)]
            P = indices[1:(l - 1)]
            PᵀU = U[P, :]
            uₗ = basis[:, l]
            Pᵀuₗ = uₗ[P, :]
            c = vec(PᵀU \ Pᵀuₗ)
            mul!(r, U, c)
            @. r = abs(uₗ - r)
            indices[l] = argmax(r)
        end
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
    @set! sys.name = name
    
    # handle ODESystem.substitutions
    # https://github.com/SciML/ModelingToolkit.jl/issues/1754
    sys = tearing_substitution(sys)

    iv = sys.iv # the single independent variable
    D = Differential(iv)

    V = pod_basis
    pod_dim = size(V, 2) # the dimension of POD basis
    ŷ = @variables ŷ(iv)[1:pod_dim] # new variables from POD reduction
    @set! sys.states = ŷ

    dvs = sys.states # dependent variables
    deqs, eqs = get_deqs(sys) # split eqs into differential and non-differential equations

    rhs = Symbolics.rhss(deqs)
    # a sparse matrix of coefficients for the linear part,
    # a vector of constant terms and a vector of nonlinear terms about dvs
    A, g, F = linear_terms(rhs, dvs)

    pod_eqs = Symbolics.scalarize(dvs .~ V * ŷ)
    @set! sys.observed = [sys.observed; pod_eqs]

    inv_dict = Dict(ŷ .=> V' * dvs) # reduced vars to orignial vars
    @set! sys.defaults = merge(sys.defaults, inv_dict)

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

    @set! sys.eqs = [Symbolics.scalarize(deqs); eqs]
end
