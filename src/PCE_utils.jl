# getting independent variables
function get_independent_vars(var)
    return []
end
function get_independent_vars(var::Symbolics.Term) where T
    if operation(var) isa Symbolics.Sym
        return arguments(var)
    else
        return reduce(vcat, get_independent_vars(arguments(var)))
    end
end
function get_independent_vars(var::Num)
    return get_independent_vars(Symbolics.unwrap(var))
end
function get_independent_vars(vars::AbstractVector)
    return [get_independent_vars(var) for var in vars]
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
        return term, 1.0
    end
end

function split_term(term::T, vars) where T <: Union{Symbolics.Term, Symbolics.Sym}
    if term in vars
        return 1.0, term
    else
        return term, 1.0
    end
end 

function extract_coeffs(expr::Symbolics.Add, vars::Set)
    coeffs = Dict()
    for term in keys(expr.dict)
        num_coeff = expr.dict[term]
        var_coeff, mono = split_term(term, vars)
        try 
            coeffs[mono] += num_coeff*var_coeff
        catch
            coeffs[mono] = num_coeff*var_coeff
        end
    end 
    return coeffs
end

function extract_coeffs(expr::Symbolics.Mul, vars::Set)
    coeff, mono = split_term(expr, vars)
    return Dict(mono => coeff)
end

extract_coeffs(expr::Num, vars::Set) = extract_coeffs(Symbolics.unwrap.(expr), vars)
extract_coeffs(expr::Num, vars::AbstractArray) = extract_coeffs(Symbolics.unwrap.(expr), vars)
extract_coeffs(expr, vars::AbstractArray{<:Num}) = extract_coeffs(expr, Symbolics.unwrap.(vars))
extract_coeffs(expr, vars::AbstractArray) = extract_coeffs(expr, Set(vars))

function get_basis_indices(mono::Symbolics.Mul)
    basis_indices = Int[]
    for (term, pow) in mono.dict
        append!(basis_indices, (arguments(term)[end] - 1)*ones(Int, pow))
    end
    return basis_indices
end
function get_basis_indices(mono::Symbolics.Term)
    return [arguments(mono)[end] - 1]
end
function get_basis_indices(mono::Symbolics.Pow)
    return (arguments(mono.base)[end] - 1)*ones(Int, mono.exp)
end
function get_basis_indices(mono::Num)
    return get_basis_indices(Symbolics.unwrap(mono))
end