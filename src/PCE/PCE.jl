using PolyChaos, Symbolics, ModelingToolkit, LinearAlgebra

export PCE, moment_equations, pce_galerkin, mean, var

include("PCE_utils.jl")

"""
$(SIGNATURES)

`PCE` object for symbolic representation of a dense multivariate polynomial chaos expansion
of a given set of states ``x`` in terms of a given set of uncertain parameters ``p``:
```math
    x = ∑ᵢ zᵢ ζᵢ(p). 
```
Here ``x`` denotes the states of the PCE, ``p`` the parameters, ``zᵢ`` refers to the ``i``th moments and ``ζᵢ`` to the
``i``th basis function. 

# Fields
- `states`: Vector of states (symbolic variables) representing the state of the PCE.
- `parameters`: `Vector` of parameters (symbolic variables) being expanded.
- `uni_basis`: `Vector` of `Pair`s mapping parameters to a `PolyChaos.AbstractOrthoPoly` representing 
                 the basis in which the parametric dependence is expanded.
- `tensor_basis`: `TensorProductOrthoPoly` representing the tensorproduct-based multi-variate basis underpinning the PCE
- `sym_basis`: `Vector` of symbolic variables representing the basis functions: ``[ζ₁, …, ζₘ]``.
- `ansatz`: `Vector` of `Pair`s mapping state to corresponding PCE ansatz
- `moments`: `Vector` of `Vector`s carrying the moments for each state.
"""
struct PCE
    states::Vector{<:Num}
    parameters::Vector{<:Num}
    uni_basis::Vector{Pair{<:Num, <:Union{AbstractOrthoPoly, AbstractCanonicalOrthoPoly}}}
    tensor_basis::TensorProductOrthoPoly
    sym_basis::Vector{<:Num}
    ansatz::Vector{Pair{<:Num, <:Num}}
    moments::Vector{Vector{<:Num}}
end

"""
$(SIGNATURES)
    
Create `PCE` object from a `Vector` of states (symbolic variables) and a `Vector` of `Pair`s mapping
the parameters to the corresponding `PolyChaos.AbstractOrthoPoly` basis used for expansion.
"""
function PCE(states, uni_basis::AbstractVector{<:Pair})
    # to deal with symbolic arrays
    states = collect(states)

    parameters = [p for (p, op) in uni_basis]
    ops = [op for (p, op) in uni_basis]

    @assert all(deg(op) > 0 for op in ops) "Every basis considered must at least include the linear function"

    tensor_basis = TensorProductOrthoPoly(ops)
    n_basis = size(tensor_basis.ind, 1)
    n_states = length(states)

    @variables ζ(parameters...)[1:n_basis]
    sym_basis = collect(ζ)

    moments = Vector{Num}[]
    for (i, state) in enumerate(collect(states))
        ind_vars = get_independent_vars(state)
        new_vars = if isempty(ind_vars)
            Symbolics.variables(:z, i:i, 1:n_basis)
        else
            map(z -> z(ind_vars...),
                Symbolics.variables(:z, i:i, 1:n_basis; T = Symbolics.FnType))
        end
        push!(moments, vec(new_vars))
    end
    ansatz = [states[i] => sum(moments[i][j] * sym_basis[j] for j in 1:n_basis)
              for i in 1:n_states]
    return PCE(states, parameters, uni_basis, tensor_basis, sym_basis, ansatz, moments)
end
function (pce::PCE)(moment_vals, parameter_vals::AbstractVector)
    basis = evaluate(parameter_vals, pce.tensor_basis)
    return [dot(moments, basis) for moments in moment_vals]
end

# 1. apply PCE ansatz
"""
$(TYPEDSIGNATURES)

Generate linear PCEs for the uncertain parameters.
"""
function generate_parameter_pce(pce::PCE)
    par_dim = length(pce.parameters)
    par_pce = Vector{Pair{eltype(pce.parameters), eltype(pce.sym_basis)}}(undef, par_dim)
    for (i, basis) in enumerate(pce.uni_basis)
        p, op = basis
        par_pce[i] = p => pce.sym_basis[i + 1] + op.α[1]
    end
    return par_pce
end

"""
$(TYPEDSIGNATURES)

Substitute parameters shared between a set of symbolic `eqs` and the PCE `pce` for the corresponding linear PCEs.  
"""
function substitute_parameters(eqs::AbstractVector, pce::PCE)
    par_pce = generate_parameter_pce(pce)
    subs_eqs = [substitute(eq, par_pce) for eq in eqs]
    return subs_eqs
end

"""
$(TYPEDSIGNATURES)

Substitute PCE Ansatz defined in `pce` into a set of symbolic equations `eqs`.
"""
function substitute_pce_ansatz(eqs::AbstractVector, pce::PCE)
    subs_eqs = [expand(expand(substitute(eq, pce.ansatz))) for eq in eqs]
    return subs_eqs
end

"""
$(TYPEDSIGNATURES)

Apply PCE ansatz defined in `pce` to a given set of symbolic equations `eqs`.
"""
function apply_ansatz(eqs::AbstractVector, pce::PCE)
    return substitute_pce_ansatz(substitute_parameters(eqs, pce), pce)
end

# 2. extract PCE expansion coeffs
"""
$(TYPEDSIGNATURES)

Given a set of symbolic equations `eqs` involving the basis functions of `pce`, 
extract monomials of the basis functions and the corresponding coeffiecients.

# Returns
`Vector` of `Dict`s mapping monomial of basis functions to its coefficient in the individual equations.
"""
function extract_basismonomial_coeffs(eqs::AbstractVector, pce::PCE)
    basismonomial_coeffs = [extract_coeffs(eq, pce.sym_basis) for eq in eqs]
    basismonomial_indices = []
    for coeffs in basismonomial_coeffs
        union!(basismonomial_indices,
               [mono => get_basis_indices(mono) for mono in keys(coeffs)])
    end
    return basismonomial_coeffs, basismonomial_indices
end

# 3. compute inner products
"""
$(TYPEDSIGNATURES)

Evaluate scalar products between all basis functions in `pce` and 
basis monomials as characterized by `mono_indices`.
"""
function eval_scalar_products(mono_indices, pce::PCE)
    uni_degs = [deg(op) for op in pce.tensor_basis.uni]
    max_degs = uni_degs
    for (mono, id) in mono_indices
        max_degs = max.(max_degs, vec(sum(pce.tensor_basis.ind[id .+ 1, :], dims = 1)))
    end
    quad_deg = max.(ceil.(Int, 0.5 * (max_degs + uni_degs .+ 1)))

    integrators = map((uni, deg) -> bump_degree(uni, deg), pce.tensor_basis.uni, quad_deg)
    scalar_products = Dict()
    for k in 1:dim(pce.tensor_basis)
        scalar_products[k] = Dict(mono => computeSP(vcat(id, k - 1), pce.tensor_basis,
                                                    integrators)
                                  for (mono, id) in mono_indices)
    end
    return scalar_products
end

# 4. Galerkin projection
"""
$(TYPEDSIGNATURES)

perform Galerkin projection of polynomial expressions characterized by `Dict`s mapping
basis monomials to coefficients. 
"""
function galerkin_projection(bm_coeffs::Vector{<:Dict}, scalar_products::Dict,
                             pce::PCE)
    projected_eqs = []
    scaling_factors = computeSP2(pce.tensor_basis)
    for i in eachindex(bm_coeffs)
        eqs = []
        for k in 1:dim(pce.tensor_basis)
            push!(eqs,
                  sum(bm_coeffs[i][mono] * scalar_products[k][mono]
                      for mono in keys(bm_coeffs[i])))
        end
        push!(projected_eqs, eqs ./ scaling_factors)
    end
    return projected_eqs
end

# 5. combine everything
"""
$(TYPEDSIGNATURES)

perform Galerkin projection onto the `pce`.
"""
function pce_galerkin(eqs::AbstractVector, pce::PCE)
    expanded_eqs = apply_ansatz(eqs, pce)
    basismono_coeffs, basismono_idcs = extract_basismonomial_coeffs(expanded_eqs, pce)
    scalar_products = eval_scalar_products(basismono_idcs, pce)
    projected_eqs = galerkin_projection(basismono_coeffs, scalar_products, pce)
    return projected_eqs
end

# 6. high-level interface
# 6a. apply pce to explicit ODE
"""
$(TYPEDSIGNATURES)

Generate moment equations of an `ODESystem` from a given `PCE`-Ansatz via Galerkin projection.
"""
function moment_equations(sys::ODESystem, pce::PCE)
    eqs = [eq.rhs for eq in equations(sys)]
    projected_eqs = pce_galerkin(eqs, pce)
    moment_eqs = reduce(vcat, projected_eqs)
    iv = ModelingToolkit.get_iv(sys)
    params = setdiff(parameters(sys), pce.parameters)
    D = Differential(iv)
    moments = reduce(vcat, pce.moments)
    name = Symbol(String(nameof(sys)) * "_pce")
    pce_system = ODESystem([D(moments[i]) ~ moment_eqs[i] for i in eachindex(moments)],
                           iv, moments, params, name = name)

    n_moments = dim(pce.tensor_basis)
    n_states = length(states(sys))
    pce_eval = function (moment_vals, parameter_values)
        shape_state = [moment_vals[(i * n_moments + 1):((i + 1) * n_moments)]
                       for i in 0:(n_states - 1)]
        return pce(shape_state, parameter_values)
    end
    return pce_system, pce_eval
end

# 6b. apply pce to implicit & mass-matrix ODE/DAEs

# 6c. apply pce to algebraic equations

# 6d. apply pce to control problems

# 6e. ? 

# ToDo:
# better support to evaluate the PCE
# in particular => make evaluation of means, variances, etc evaluable from the object itself upon specification of the moment values
# hinderance -> how do you provide the moments in a convenient format?
