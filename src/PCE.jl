using PolyChaos, Symbolics, ModelingToolkit, LinearAlgebra

export PCE 

include("PCE_utils.jl")
# for now only consider tensor grid bases
# with degree equal across all bases
# need to adjust in PolyChaos
struct PCE 
    states # states
    parameters # vector of parameters being expanded
    bases # vector of pairs: p (symbolic variable) => polynomial basis (PolyChaos)
    bases_dict # dictionary generated from bases
    sym_basis # vector of basis functions (symbolic variables) !![indexing as generated by PolyChaos]!!
    pc_basis # polychaos basis object 
    sym_to_pc # dictionary mapping symbolic to pc basis 
    pc_to_sym # dictionary mapping pc to symbolic basis 
    ansatz # vector of pairs: x(t,p) => ∑ᵢ cᵢ(t)ξᵢ(p)
    moments # matrix (?) of symbolic variables: cᵢ(t)
end
function PCE(states, bases::AbstractVector{<:Pair})
    # to deal with symbolic arrays
    states = collect(states)

    bases_dict = Dict(bases)
    parameters = [p for (p, op) in bases]
    ops = [op for (p, op) in bases]
    degs = [deg(op) for op in ops]
    min_deg = minimum(degs)
    if !(allequal(degs))
        @warn "Currently only bases with identical degrees are supported. Proceed with minimum common degree = $min_deg"
    end
    pc_basis = MultiOrthoPoly(ops, min_deg)
    n_basis = size(pc_basis.ind,1)
    n_states = length(states)
    
    @variables ζ(parameters...)[1:n_basis]
    sym_basis = collect(ζ)

    sym_to_pc = Dict(ζ[i] => pc_basis.ind[i,:] for i in eachindex(ζ))
    pc_to_sym = Dict(val => key for (val, key) in sym_to_pc)

    moments = []
    for (i,state) in enumerate(collect(states))
        moment_name = Symbol("z" * Symbolics.map_subscripts(i))
        ind_vars = get_independent_vars(state)
        if isempty(ind_vars)
            pce_coeffs = @variables $(moment_name)[1:n_basis]
        else
            pce_coeffs = @variables $(moment_name)(ind_vars...)[1:n_basis]
        end
        push!(moments, collect(pce_coeffs[1]))
    end
    ansatz = [states[i] => sum(moments[i][j]*sym_basis[j] for j in 1:n_basis) for i in 1:n_states]
    return PCE(states, parameters, bases, bases_dict, sym_basis, pc_basis, sym_to_pc, pc_to_sym, ansatz, moments)
end
function (pce::PCE)(moment_vals, parameter_vals::AbstractMatrix)
    # wasteful => should implement my own version of this
    # this evaluates each polynomial via recurrence relation from scratch
    # can reuse many results. 
    # fine for now. 
    basis = evaluate(parameter_vals, pce.pc_basis)
    return [dot(moments,basis) for moments in moment_vals]
end
function (pce::PCE)(moment_vals, parameter_vals::AbstractVector)
    return pce(moment_vals, reshape(parameter_vals,1,length(parameter_vals)))
end
function (pce::PCE)(moment_vals, parameter_vals::Number)
    return pce(moment_vals, reshape([parameter_vals],1,1))
end

# 1. apply PCE ansatz
function generate_parameter_pce(pce::PCE)
    par_dim = length(pce.parameters)
    par_pce = Vector{Pair{eltype(pce.parameters), eltype(pce.sym_basis)}}(undef, par_dim)
    for (i, bases) in enumerate(pce.bases)
        p, op = bases
        par_pce[i] = p => pce.sym_basis[i+1] + op.α[1]
    end
    return par_pce
end
function substitute_parameters(eqs::AbstractVector, pce::PCE)
    par_pce = generate_parameter_pce(pce)
    subs_eqs = [substitute(eq, par_pce) for eq in eqs]
    return subs_eqs
end
function substitute_pce_ansatz(eqs::AbstractVector, pce::PCE)
    subs_eqs = [expand(expand(substitute(eq, pce.ansatz))) for eq in eqs]
    return subs_eqs
end
function apply_ansatz(eqs::AbstractVector, pce::PCE)
    return substitute_pce_ansatz(substitute_parameters(eqs, pce), pce)
end

# 2. extract PCE expansion coeffs 
function extract_basismonomial_coeffs(eqs::AbstractVector, pce::PCE)
    basismonomial_coeffs = [extract_coeffs(eq, pce.sym_basis) for eq in eqs]
    basismonomial_indices = []
    for coeffs in basismonomial_coeffs
        union!(basismonomial_indices, [mono => get_basis_indices(mono) for mono in keys(coeffs)])
    end
    return basismonomial_coeffs, basismonomial_indices
end

# 3. compute inner products
function maximum_degree(mono_indices::AbstractVector, pce::PCE)
    max_degree = 0
    for (mono, ind) in mono_indices
        max_degree = max(max_degree, maximum(sum(ind[i]*pce.pc_basis.ind[i+1] for i in eachindex(ind))))
    end
    return max_degree
end
function eval_scalar_products(mono_indices, pce::PCE)
    max_degree = maximum_degree(mono_indices, pce)
    degree_quadrature = ceil(Int, 0.5 * (max_degree + deg(pce.pc_basis) + 1))
    integrator_pce = bump_degree(pce.pc_basis, degree_quadrature)

    scalar_products = Dict()
    for k in 1:dim(pce.pc_basis)
        scalar_products[k] = Dict([mono => computeSP([ind..., k-1], integrator_pce) for (mono, ind) in mono_indices])
    end
    return scalar_products
end

# 4. Galerkin projection

# 5. combine everything to high-level interface
