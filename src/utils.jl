using SparseArrays
using ModelingToolkit: iscall, operation

"""
$(TYPEDSIGNATURES)

Return the differential equations and other non-differential equations.

For differential equations, this function assume the derivative is the only single term at
the left hand side, which is typically the result of `ModelingToolkit.mtkcompile`.

Equations from subsystems are not included.
"""
function get_deqs(sys::ODESystem)::Tuple{Vector{Equation}, Vector{Equation}}
    eqs = ModelingToolkit.get_eqs(sys)
    deqs = Equation[]
    others = Equation[]
    for eq in eqs
        if iscall(eq.lhs) && operation(eq.lhs) isa Differential
            push!(deqs, eq)
        else
            push!(others, eq)
        end
    end
    deqs, others
end

"""
$(SIGNATURES)

Returns `true` is `expr` contains variables in `dvs` only and does not contain `iv`.

"""
function only_dvs(expr, dvs, iv)
    if isequal(expr, iv)
        return false
    elseif expr in dvs
        return true
    elseif SymbolicUtils.iscall(expr)
        args = arguments(expr)
        for arg in args
            if only_dvs(arg, dvs, iv)
                return true
            end
        end
    end
    return false
end

"""
$(SIGNATURES)

Given a vector of expressions `exprs`, variables `vars` and a single variable `iv`,
where `vars(iv)` are dependent variables of `iv`,
returns a sparse coefficient matrix `A`, other terms `g` and nonlinear terms `F`,
such that `exprs = A * vars(iv) + g(iv) + F(vars(iv))`,
where the nonlinear terms are functions of `vars` only and do not contain `iv`.

Variables in `vars` must be unique.
"""
function separate_terms(exprs::AbstractVector, vars, iv)
    vars = Symbolics.unwrap.(vars)
    exprs = Symbolics.unwrap.(exprs)
    # expand is helpful for separating terms but is harmful for generating efficient runtime functions
    exprs = expand.(exprs)
    linear_I = Int[] # row idx for sparse matrix
    linear_J = Int[] # col idx for sparse matrix
    linear_V = Float64[] # values
    idxmap = Dict(v => i for (i, v) in enumerate(vars))
    if length(idxmap) < length(vars)
        throw(ArgumentError("vars: $vars are not unique"))
    end
    vars = keys(idxmap)

    # (a function used to avoid code duplication)
    # when a linear term is identified, add the indices and value to the sparse matrix
    @inline function push_sparse_coeff(row_index, term, value)
        push!(linear_I, row_index)
        push!(linear_J, idxmap[term])
        push!(linear_V, value)
        nothing
    end

    other_terms = similar(exprs, Num) # create a vector of the same size
    other_terms .= 0 # manually set to Int 0 because `Num` doesn't have a corresponding zero
    nonlinear_terms = similar(exprs, Num)
    nonlinear_terms .= 0

    # check if the expr is a nolinear term about vars only
    # and add it to the corresponding collection
    @inline function other_nonlinear(i, expr)
        # expr is nonlinear if it contains vars only
        if only_dvs(expr, vars, iv)
            nonlinear_terms[i] += expr
        else
            other_terms[i] += expr
        end
        nothing
    end

    for (i, expr) in enumerate(exprs)
        if expr isa Number # just a number, e.g. Int, Float64
            other_terms[i] = expr
        elseif expr in vars # expr is a variables in vars
            push_sparse_coeff(i, expr, 1)
        elseif SymbolicUtils.ismul(expr) && length(expr.dict) == 1
            base, exp = first(expr.dict)
            if base in vars && exp == 1 # a var with a coeff
                push_sparse_coeff(i, base, expr.coeff)
            else
                other_nonlinear(i, expr)
            end
        elseif SymbolicUtils.isadd(expr)
            other_terms[i] += expr.coeff
            for (term, coeff) in expr.dict
                if term in vars
                    push_sparse_coeff(i, term, coeff)
                else
                    other_nonlinear(i, term * coeff)
                end
            end
        else
            other_nonlinear(i, expr)
        end
    end
    linear = sparse(linear_I, linear_J, linear_V, length(exprs), length(vars))
    return linear, other_terms, nonlinear_terms
end