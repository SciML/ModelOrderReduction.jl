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

Reduce a `ModelingToolkit.ODESystem` using the Proper Orthogonal Decomposition (POD) with
the Discrete Empirical Interpolation Method (DEIM).

`snapshot` should be a matrix with the data of each time instance as a column.

The LHS of equations in `sys` are all assumed to be 1st order derivatives. Use
`ModelingToolkit.ode_order_lowering` to transform higher order ODEs before applying DEIM.

`sys` is assumed to have no internal systems. End users are encouraged to call
`ModelingToolkit.structural_simplify` beforehand.

The POD basis used for DEIM interpolation is obtained from the snapshot matrix of the
nonlinear terms, which is computed by executing the runtime-generated function for
nonlinear expressions.
"""
function deim(sys::ODESystem, snapshot, pod_dim::Integer; deim_dim::Integer = pod_dim,
              name::Symbol = Symbol(nameof(sys), :_deim))::ODESystem
    @set! sys.name = name

    # handle ODESystem.substitutions
    # https://github.com/SciML/ModelingToolkit.jl/issues/1754
    sys = tearing_substitution(sys)

    iv = ModelingToolkit.get_iv(sys) # the single independent variable
    D = Differential(iv)
    dvs = ModelingToolkit.get_states(sys) # dependent variables

    pod_reducer = POD(snapshot, pod_dim)
    reduce!(pod_reducer, TSVD())
    V = pod_reducer.rbasis # POD basis

    var_name = gensym(:ŷ)
    ŷ = (@variables $var_name(iv)[1:pod_dim])[1]
    @set! sys.states = Symbolics.value.(Symbolics.scalarize(ŷ)) # new variables from POD
    ModelingToolkit.get_var_to_name(sys)[Symbolics.getname(ŷ)] = Symbolics.unwrap(ŷ)

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
    @set! sys.defaults = merge(ModelingToolkit.defaults(sys), inv_dict)

    pod_dict = Dict(eq.lhs => eq.rhs for eq in pod_eqs) # original vars to reduced vars

    # generate an in-place function from the symbolic expression of the nonlinear functions
    F_expr = build_function(F, dvs; expression = Val{false})[2]
    F_func! = eval(F_expr)
    nonlinear_snapshot = similar(snapshot) # snapshot matrix of nonlinear terms
    for i in 1:size(snapshot, 2) # iterate through time instances
        F_func!(view(nonlinear_snapshot, :, i), view(snapshot, :, i))
    end

    deim_reducer = POD(nonlinear_snapshot, deim_dim)
    reduce!(deim_reducer, TSVD())
    U = deim_reducer.rbasis # DEIM projection basis

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
end
