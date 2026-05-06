unwrap(x) = Symbolics.value(x)

isnearinteger(x; tol = 1.0e-6) = x isa Number && abs(x - round(x)) <= tol

isleaf_expr(expr, iv::Num) = !iscall(expr) || isequal(iv, arguments(unwrap(expr))[1])

function lhs_to_rhs_dict(eqns)
    return Dict([eq.lhs => eq.rhs for eq in eqns])
end

struct VariablesHolder
    state_variables::Vector{Num}
    num_vars_orig::Int
    base_name::String
    start_with::Int
    function VariablesHolder(vars, new_var_base_name = "z_", start_with = 0)
        return new(vars, length(vars), new_var_base_name, start_with)
    end
end

function create_variable!(vars::VariablesHolder, iv::Num)
    new_index = length(vars.state_variables) + vars.start_with - vars.num_vars_orig
    new_variable = Symbolics.scalarize(@variables $(Symbol(vars.base_name, new_index))(iv))[1]
    push!(vars.state_variables, new_variable)
    return new_variable
end

function delete_variable!(vars::VariablesHolder, var::Num)
    pop!(vars.state_variables)
    return nothing
end

"""
Normalizes a symbolic expression by recursively rewriting quotients, powers,
products, and sums into a more canonical form. Roughly speaking, divide and minus 
operations are moved pushed to the start of the expression tree.

The normalization procedure separates numerator, denominator, and sign data, then
rebuilds the expression. If `absolute=true`, the final sign is discarded. For example,
`normalize_symbolic_function(-y / x, eqsys, true)` returns `y / x`.
"""
function normalize_util(expr, iv::Num; new_normalize = normalize_util, absolute = false, eqsys = nothing)
    function helper(unwrap_expr)
        if unwrap_expr isa Number && isless(unwrap_expr, 0)
            return (-unwrap_expr, 1, -1)
        elseif isleaf_expr(unwrap_expr, iv)
            return (unwrap_expr, 1, 1)
        elseif operation(unwrap_expr) == sqrt
            base = arguments(unwrap_expr)[1]
            num_of_base, denom_of_base, sign_of_base = helper(base)
            return ((num_of_base * sign_of_base)^(1 // 2), isequal(1, denom_of_base) ? 1 : denom_of_base^(1 // 2), 1)
        elseif operation(unwrap_expr) == /
            num, denom = unwrap.(arguments(unwrap_expr))
            num_of_num, denom_of_num, sign_of_num = helper(num)
            num_of_denom, denom_of_denom, sign_of_denom = helper(denom)
            return (num_of_num * denom_of_denom, denom_of_num * num_of_denom, sign_of_num * sign_of_denom)
        elseif operation(unwrap_expr) == ^
            base, exponent = unwrap.(arguments(unwrap_expr))
            num_of_base, denom_of_base, sign_of_base = helper(base)
            if ! isnothing(eqsys)
                #Polynomialization
                norm_exp = unwrap(new_normalize(exponent, eqsys))
            else
                #Quadratization
                norm_exp = unwrap(new_normalize(exponent, iv; new_normalize = new_normalize))
            end
            if norm_exp isa Number && isnearinteger(norm_exp)
                norm_exp = Integer(round(norm_exp))
                return (num_of_base^norm_exp, isequal(1, denom_of_base) ? 1 : denom_of_base^norm_exp, (round(norm_exp) % 2 == 0) ? 1 : sign_of_base)
            else
                return ((sign_of_base * num_of_base)^norm_exp, isequal(1, denom_of_base) ? 1 : denom_of_base^norm_exp, 1)
            end
        elseif operation(unwrap_expr) == *
            numdenom_args = [helper(unwrap(arg)) for arg in arguments(unwrap_expr)]
            return (prod(tup[1] for tup in numdenom_args), prod(tup[2] for tup in numdenom_args), prod(tup[3] for tup in numdenom_args))
        elseif operation(unwrap_expr) == -
            numdenom_args = [helper(unwrap(arg)) for arg in arguments(unwrap_expr)]
            sign = (numdenom_args[1][3] == -1) ? -1 : 1
            return (numdenom_args[1][1] / numdenom_args[1][2] * numdenom_args[1][3] * sign - numdenom_args[2][1] / numdenom_args[2][2] * numdenom_args[2][3] * sign, 1, sign)
        elseif operation(unwrap_expr) == +
            numdenom_args = [helper(unwrap(arg)) for arg in arguments(unwrap_expr)]
            sign = numdenom_args[1][3] == 1 ? 1 : -1
            return (sum(tup[1] / tup[2] * tup[3] * sign for tup in numdenom_args), 1, sign)
        else
            norm_args = []
            for arg in arguments(unwrap_expr)
                if ! isnothing(eqsys)
                    #Polynomialization
                    push!(norm_args, new_normalize(arg, eqsys))
                else
                    #Quadratization
                    push!(norm_args, new_normalize(arg, iv; new_normalize = new_normalize))
                end
            end
            return (operation(unwrap_expr)(norm_args...), 1, 1)
        end
    end
    tup = Num.(helper(unwrap(expr)))
    sign = absolute ? 1 : tup[3]
    return sign * tup[1] / tup[2]
end

function remove_negative_powers(expr)
    expr = unwrap(expr)

    if !iscall(expr)
        return Num(expr)
    end

    op = operation(expr)
    args = arguments(expr)

    if op == ^
        base, exponent = args
        exponent = unwrap(exponent)

        new_base = remove_negative_powers(Num(base))

        if exponent isa Number && exponent < 0
            return Num(1 / unwrap(new_base)^(-exponent))
        else
            return Num(unwrap(new_base)^exponent)
        end
    end

    new_args = [remove_negative_powers(Num(arg)) for arg in args]
    return Num(op(unwrap.(new_args)...))
end

function laurent_expand(expr, normalize_func::Function)
    function helper(expr, denom)
        expr = unwrap(normalize_func(expr))
        expr = unwrap(expand(expr))
        if iscall(expr) && !(operation(expr) in (+, -))
            expr = unwrap(expand(expr))
        end
        if !iscall(expr) || operation(expr) == *
            return expr / denom
        end
        terms = [expr]
        if iscall(expr) && operation(expr) == +
            terms = arguments(expr)
        elseif iscall(expr) && operation(expr) == -
            terms = [arguments(expr)[1], -arguments(expr)[2]]
        end
        tot = 0
        for term in terms
            if iscall(term) && operation(term) == /
                unwrap_num, unwrap_denom = unwrap.(arguments(term))
                tot += Num(helper(unwrap_num, denom * unwrap_denom))
            else
                tot += term / denom
            end
        end
        return tot
    end
    return helper(expr, 1)
end
