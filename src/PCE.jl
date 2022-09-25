using PolyChaos, Symbolics, ModelingToolkit, LinearAlgebra

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


