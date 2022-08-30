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

    iv = ModelingToolkit.get_iv(sys) # the single independent variable
    D = Differential(iv)
    dvs = ModelingToolkit.get_states(sys) # dependent variables

    V = pod_basis
    pod_dim = size(V, 2) # the dimension of POD basis
    @variables ŷ(iv)[1:pod_dim] # a symbolic array
    @set! sys.states = Symbolics.value.(Symbolics.scalarize(ŷ)) # new variables from POD
    sys.var_to_name[Symbolics.getname(ŷ)] = Symbolics.unwrap(ŷ)

    deqs, eqs = get_deqs(sys) # split eqs into differential and non-differential equations
    rhs = Symbolics.rhss(deqs)
    # a sparse matrix of coefficients for the linear part,
    # a vector of constant terms and a vector of nonlinear terms about dvs
    A, g, F = linear_terms(rhs, dvs)

    pod_eqs = Symbolics.scalarize(dvs .~ V * ŷ)
    old_observed = ModelingToolkit.get_observed(sys)
    fullstates = [map(eq -> eq.lhs, old_observed); dvs; ModelingToolkit.get_states(sys)]
    new_observed = [old_observed; pod_eqs]
    new_sorted_observed = ModelingToolkit.topsort_equations(new_observed, fullstates)
    @set! sys.observed = new_sorted_observed

    inv_dict = Dict(Symbolics.scalarize(ŷ .=> V' * dvs)) # reduced vars to orignial vars
    @set! sys.defaults = merge(sys.defaults, inv_dict)

    pod_dict = Dict(eq.lhs => eq.rhs for eq in pod_eqs) # original vars to reduced vars

    U = @view deim_basis[:, 1:deim_dim] # DEIM projection basis

    indices = deim_interpolation_indices(U) # DEIM interpolation indices
    # the DEIM projector (not DEIM basis) satisfies
    # F(original_vars) ≈ projector * F(pod_basis * reduced_vars)[indices]
    projector = ((@view U[indices, :])' \ (U' * V))'
    temp = substitute.(F[indices], (pod_dict,))
    F̂ = projector * temp # DEIM approximation for nonlinear func F

    Â = V' * A * V
    ĝ = V' * g
    deqs = D.(ŷ) ~ Â * ŷ + ĝ + F̂

    @set! sys.eqs = [Symbolics.scalarize(deqs); eqs]

    ODESystem(sys.eqs, sys.iv, sys.states, sys.ps;
              observed = ModelingToolkit.get_observed(sys), name = sys.name,
              defaults = sys.defaults)
end
