"""
$(TYPEDSIGNATURES)

Return the differential equations and other non-differential equations.

For differential equations, this function assume the derivative is the only single term at
the left hand side, which is typically the result of `ModelingToolkit.structural_simplify`.

Equations from subsystems are not included.
"""
function get_deqs(sys::ODESystem)::Tuple{Vector{Equation}, Vector{Equation}}
    eqs = ModelingToolkit.get_eqs(sys)
    deqs = Equation[]
    others = Equation[]
    for eq in eqs
        if istree(eq.lhs) && operation(eq.lhs) isa Differential
            push!(deqs, eq)
        else
            push!(others, eq)
        end
    end
    deqs, others
end

"""
$(SIGNATURES)

Given a vector of expressions `exprs` and variables `vars`,
returns a sparse coefficient matrix `A`, constant terms `c` and nonlinear terms `n`,
such that `exprs = A * vars + c + n`,
where the constant terms do not contain any variables in `vars`.

Variables in `vars` must be unique.
"""
function linear_terms(exprs::AbstractVector, vars)
    vars = Symbolics.unwrap.(vars)
    exprs = Symbolics.unwrap.(exprs)
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

    const_terms = similar(exprs, Num) # create a vector of the same size
    const_terms .= 0 # manually set to Int 0 because `Num` doesn't have a corresponding zero
    nonlinear_terms = similar(exprs, Num)
    nonlinear_terms .= 0

    # check if the expr is a constant or nolinear term about vars
    # and add it to the corresponding collection
    @inline function const_nonlinear(i, expr)
        # expr is nonlinear if it contains any variable in vars
        if Symbolics.has_vars(expr, vars)
            nonlinear_terms[i] += expr
        else # expr is constant if it doesn't have vars
            const_terms[i] += expr
        end
        nothing
    end

    for (i, expr) in enumerate(exprs)
        if expr isa Number # just a number, e.g. Int, Float64
            const_terms[i] = expr
        elseif expr in vars # expr is a variables in vars
            push_sparse_coeff(i, expr, 1)
        elseif SymbolicUtils.ismul(expr) && length(expr.dict) == 1
            base, exp = first(expr.dict)
            if base in vars && exp == 1 # a var with a coeff
                push_sparse_coeff(i, base, expr.coeff)
            else
                const_nonlinear(i, expr)
            end
        elseif SymbolicUtils.isadd(expr)
            const_terms[i] += expr.coeff
            for (term, coeff) in expr.dict
                if term in vars
                    push_sparse_coeff(i, term, coeff)
                else
                    const_nonlinear(i, term * coeff)
                end
            end
        else
            const_nonlinear(i, expr)
        end
    end
    linear = sparse(linear_I, linear_J, linear_V, length(exprs), length(vars))
    return linear, const_terms, nonlinear_terms
end

# utiltiites to extracting coefficients of a polynomial in monomial basis in variables `vars`
function split_term(term::Symbolics.Mul, vars)
    coeff = term.coeff
    mono = Num(1.0)
    for var in keys(term.dict)
        if var in vars
            mono *= var^term.dict[var]
        else
            coeff *= var^term.dict[var]
        end
    end
    return coeff, mono
end
function split_term(term::Symbolics.Pow, vars)
    if term.base in vars
        return 1.0, term
    else
        return term, Val(1)
    end
end
function split_term(term::T, vars) where {T <: Union{Symbolics.Term, Symbolics.Sym}}
    if term in vars
        return 1.0, term
    else
        return term, Val(1)
    end
end

function extract_coeffs(expr::Symbolics.Add, vars::Set)
    coeffs = Dict()
    if !iszero(expr.coeff)
        coeffs[Val(1)] = expr.coeff
    end
    for term in keys(expr.dict)
        num_coeff = expr.dict[term]
        var_coeff, mono = split_term(term, vars)
        try
            coeffs[mono] += num_coeff * var_coeff
        catch
            coeffs[mono] = num_coeff * var_coeff
        end
    end
    return coeffs
end
function extract_coeffs(expr::T, vars::Set) where {T <: Union{Symbolics.Sym, Symbolics.Mul}}
    coeff, mono = split_term(expr, vars)
    return Dict(mono => coeff)
end
extract_coeffs(expr::Num, vars::Set) = extract_coeffs(Symbolics.unwrap.(expr), vars)
function extract_coeffs(expr::Num, vars::AbstractArray)
    extract_coeffs(Symbolics.unwrap.(expr), vars)
end
function extract_coeffs(expr, vars::AbstractArray{<:Num})
    extract_coeffs(expr, Symbolics.unwrap.(vars))
end
extract_coeffs(expr, vars::AbstractArray) = extract_coeffs(expr, Set(vars))
extract_coeffs(expr::Number, vars::Set) = Dict(Val(1) => expr)
