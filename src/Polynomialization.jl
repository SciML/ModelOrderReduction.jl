using SymbolicUtils
using Symbolics
using ModelingToolkit

# State / bookkeeping objects

struct VariablesHolder
    state_variables::Vector{Num}
    num_vars_orig::Int
    base_name::String
    start_with::Int
    function VariablesHolder(vars, new_var_base_name = "w_", start_new_vars_with = 0)
        return new(vars, length(vars), new_var_base_name, start_new_vars_with)
    end
end

function create_variable!(vars::VariablesHolder, iv::Num)
    new_index = length(vars.state_variables) - vars.num_vars_orig + vars.start_with
    new_variable = Symbolics.scalarize(@variables $(Symbol(vars.base_name, new_index))(iv))[1]
    push!(vars.state_variables, new_variable)
    return new_variable
end

function delete_variable!(vars::VariablesHolder, var::Num)
    pop!(vars.state_variables)
    return nothing
end

mutable struct PolynomializationCache
    version::Int

    normalize_cache::Dict{Tuple{Any, Bool}, Num}
    is_laurent_cache::Dict{Any, Bool}
    is_laurent_denom_cache::Dict{Any, Bool}
    try_polynomialize_cache::Dict{Tuple{Any, Int}, Tuple{Num, Bool}}
    find_nonpoly_cache::Dict{Any, Set{Any}}

    substitute_known_cache::Dict{Tuple{Any, Int}, Any}

    lhs_to_rhs_cache::Dict{Int, Dict{Any, Any}}
    rhs_to_lhs_cache::Dict{Int, Dict{Any, Any}}

    sub_rhs_index::Dict{Any, Int}
    sub_power_index::Dict{Tuple{Any, Any}, Int}
    sub_reciprocal_power_index::Dict{Tuple{Any, Any}, Int}

    sub_index_history::Vector{Vector{Tuple{Symbol, Any, Bool, Any}}}

    expr_depth_cache::Dict{Any, Int}
    expr_size_cache::Dict{Any, Int}

    function PolynomializationCache()
        return new(
            0,

            Dict{Tuple{Any, Bool}, Num}(),
            Dict{Any, Bool}(),
            Dict{Any, Bool}(),
            Dict{Tuple{Any, Int}, Tuple{Num, Bool}}(),
            Dict{Any, Set{Any}}(),

            Dict{Tuple{Any, Int}, Any}(),

            Dict{Int, Dict}(),
            Dict{Int, Dict}(),

            Dict{Any, Int}(),
            Dict{Tuple{Any, Any}, Int}(),
            Dict{Tuple{Any, Any}, Int}(),

            Vector{Vector{Tuple{Symbol, Any, Bool, Any}}}(),

            Dict{Any, Int}(),
            Dict{Any, Int}(),
        )
    end
end

struct EquationSystem
    eqns::Vector{Equation}
    substitution_equations::Vector{Equation}
    holder::VariablesHolder
    iv::Num
    cache::PolynomializationCache

    function EquationSystem(sys::System, new_var_base_name = "w_", start_new_vars_with = 0)
        eqns = equations(sys)
        variables = [Num(arguments(eqn.lhs)[1]) for eqn in eqns]
        holder = VariablesHolder(variables, new_var_base_name, start_new_vars_with)
        return new(
            eqns,
            Equation[],
            holder,
            ModelingToolkit.get_iv(sys),
            PolynomializationCache(),
        )
    end
end

function invalidate_cache!(eqsys::EquationSystem)
    eqsys.cache.version += 1

    empty!(eqsys.cache.try_polynomialize_cache)
    empty!(eqsys.cache.substitute_known_cache)

    empty!(eqsys.cache.lhs_to_rhs_cache)
    empty!(eqsys.cache.rhs_to_lhs_cache)

    return nothing
end

function trim_cache!(
        eqsys::EquationSystem;
        max_expr_entries = 20_000,
        max_small_entries = 100_000
    )
    c = eqsys.cache

    if length(c.normalize_cache) > max_expr_entries
        empty!(c.normalize_cache)
    end

    if length(c.find_nonpoly_cache) > max_expr_entries
        empty!(c.find_nonpoly_cache)
    end

    if length(c.expr_depth_cache) > max_small_entries
        empty!(c.expr_depth_cache)
    end

    if length(c.expr_size_cache) > max_small_entries
        empty!(c.expr_size_cache)
    end

    return nothing
end

function set_index_with_history!(dict::Dict, key, val, inserted_keys, kind::Symbol)
    old = get(dict, key, nothing)
    had_old = haskey(dict, key)

    dict[key] = val

    push!(inserted_keys, (kind, key, had_old, old))

    return nothing
end

function index_substitution!(eqsys::EquationSystem, i::Int, inserted_keys)
    eqn = eqsys.substitution_equations[i]
    rhs = unwrap(normalize_symbolic_function(eqn.rhs, eqsys))

    set_index_with_history!(
        eqsys.cache.sub_rhs_index,
        rhs,
        i,
        inserted_keys,
        :rhs
    )

    if iscall(rhs) && operation(rhs) == ^
        rhs_base, rhs_exp = unwrap.(arguments(rhs))
        norm_base = unwrap(normalize_symbolic_function(rhs_base, eqsys))

        key = (norm_base, rhs_exp)

        set_index_with_history!(
            eqsys.cache.sub_power_index,
            key,
            i,
            inserted_keys,
            :power
        )

        if iscall(norm_base) && operation(norm_base) == /
            num, _ = unwrap.(arguments(norm_base))
            if isequal(num, 1)
                set_index_with_history!(
                    eqsys.cache.sub_reciprocal_power_index,
                    key,
                    i,
                    inserted_keys,
                    :reciprocal_power
                )
            end
        end
    else
        key = (rhs, 1)

        set_index_with_history!(
            eqsys.cache.sub_power_index,
            key,
            i,
            inserted_keys,
            :power
        )

        if iscall(rhs) && operation(rhs) == /
            num, _ = unwrap.(arguments(rhs))
            if isequal(num, 1)
                set_index_with_history!(
                    eqsys.cache.sub_reciprocal_power_index,
                    key,
                    i,
                    inserted_keys,
                    :reciprocal_power
                )
            end
        end
    end

    return nothing
end

function undo_last_substitution_indices!(eqsys::EquationSystem)
    inserted_keys = pop!(eqsys.cache.sub_index_history)

    for (kind, key, had_old, old) in reverse(inserted_keys)
        dict =
            kind === :rhs ? eqsys.cache.sub_rhs_index :
            kind === :power ? eqsys.cache.sub_power_index :
            kind === :reciprocal_power ? eqsys.cache.sub_reciprocal_power_index :
            error("Unknown substitution index kind: $kind.")

        if had_old
            dict[key] = old
        else
            delete!(dict, key)
        end
    end

    return nothing
end


# Basic utilities

unwrap(x) = Symbolics.value(x)

isnearinteger(x; tol = 1.0e-6) = x isa Number && abs(x - round(x)) <= tol

isleaf_expr(expr, eqsys::EquationSystem) = !iscall(expr) || isequal(eqsys.iv, arguments(expr)[1])

function repeated_substitute(expr, dict; maxiters = 10)
    for _ in 1:maxiters
        newexpr = Symbolics.substitute(expr, dict)
        isequal(expr, newexpr) && return expr
        expr = newexpr
    end
    error("Substitution did not reach a fixed point after $maxiters iterations.")
end

expr_depth(x) = SymbolicUtils.istree(x) ? 1 + maximum(expr_depth, Symbolics.arguments(x)) : 0

function expr_size(x)
    x = unwrap(x)
    if !SymbolicUtils.istree(x)
        return 1
    end
    args = Symbolics.arguments(x)
    return 1 + sum(expr_size, args)
end

function expr_depth_cached(eqsys::EquationSystem, x)
    x = unwrap(x)

    if haskey(eqsys.cache.expr_depth_cache, x)
        return eqsys.cache.expr_depth_cache[x]
    end

    val = SymbolicUtils.istree(x) ?
        1 + maximum(arg -> expr_depth_cached(eqsys, arg), Symbolics.arguments(x)) :
        0

    eqsys.cache.expr_depth_cache[x] = val
    return val
end

function expr_size_cached(eqsys::EquationSystem, x)
    x = unwrap(x)

    if haskey(eqsys.cache.expr_size_cache, x)
        return eqsys.cache.expr_size_cache[x]
    end

    val = if !SymbolicUtils.istree(x)
        1
    else
        1 + sum(arg -> expr_size_cached(eqsys, arg), Symbolics.arguments(x))
    end

    eqsys.cache.expr_size_cache[x] = val
    return val
end

equalmag(x, y, eqsys::EquationSystem) = begin
    xn = normalize_symbolic_function(x, eqsys)
    yn = normalize_symbolic_function(y, eqsys)
    isequal(xn, yn)||isequal(-xn, yn)
end

equalmag_normalized(xn, yn) = isequal(xn, yn) || isequal(-xn, yn)

function rhs_to_lhs_dict(eqns)
    return Dict([eq.rhs => eq.lhs for eq in eqns])
end

function lhs_to_rhs_dict(eqns)
    return Dict([eq.lhs => eq.rhs for eq in eqns])
end

function lhs_to_rhs_dict_cached(eqsys::EquationSystem)
    v = eqsys.cache.version

    if haskey(eqsys.cache.lhs_to_rhs_cache, v)
        return eqsys.cache.lhs_to_rhs_cache[v]
    end

    dict = lhs_to_rhs_dict(eqsys.eqns)
    eqsys.cache.lhs_to_rhs_cache[v] = dict
    return dict
end

function rhs_to_lhs_dict_cached(eqsys::EquationSystem)
    v = eqsys.cache.version

    if haskey(eqsys.cache.rhs_to_lhs_cache, v)
        return eqsys.cache.rhs_to_lhs_cache[v]
    end

    dict = rhs_to_lhs_dict(eqsys.substitution_equations)
    eqsys.cache.rhs_to_lhs_cache[v] = dict
    return dict
end

# Normalization of expressions

function _normalize_uncached(expr, eqsys::EquationSystem; absolute = false)
    function helper(unwrap_expr)
        if unwrap_expr isa Number && isless(unwrap_expr, 0)
            return (-unwrap_expr, 1, -1)
        elseif isleaf_expr(unwrap_expr, eqsys)
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
            norm_exp = unwrap(normalize_symbolic_function(exponent, eqsys))
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
                push!(norm_args, normalize_symbolic_function(arg, eqsys))
            end
            return (operation(unwrap_expr)(norm_args...), 1, 1)
        end
    end
    tup = Num.(helper(unwrap(expr)))
    sign = absolute ? 1 : tup[3]
    return sign * tup[1] / tup[2]
end

"""
Normalizes a symbolic expression by recursively rewriting quotients, powers,
products, and sums into a more canonical form. Roughly speaking, divide and minus 
operations are moved pushed to the start of the expression tree.

The normalization procedure separates numerator, denominator, and sign data, then
rebuilds the expression. If `absolute=true`, the final sign is discarded. For example,
`normalize_symbolic_function(-y / x, eqsys, true)` returns `y / x`.
"""
function normalize_symbolic_function(expr, eqsys::EquationSystem; absolute = false)
    key = (unwrap(expr), absolute)

    val = get(eqsys.cache.normalize_cache, key, nothing)
    ! isnothing(val) && return val

    val = _normalize_uncached(expr, eqsys; absolute = absolute)
    eqsys.cache.normalize_cache[key] = val
    return val
end


# Laurent / polynomial classification

function is_polynomial(expr, vars::Vector{Num})
    expanded = Symbolics.expand(expr)
    _, remainder = Symbolics.polynomial_coeffs(expanded, vars)
    return Symbolics._iszero(remainder)
end

function _is_laurent_denom_uncached(expr::Num, eqsys::EquationSystem)
    unwrap_expr = unwrap(expr)
    if isleaf_expr(unwrap_expr, eqsys)
        return true
    end

    op = operation(unwrap_expr)
    args = arguments(unwrap_expr)

    if op == *
        return all(is_laurent_denom(Num(arg), eqsys) for arg in args)
    elseif op == ^
        unwrap_base, unwrap_exp = unwrap.(args)
        if !(unwrap_exp isa Number) || !isnearinteger(unwrap_exp)
            return false
        elseif unwrap_exp < -1.0e-6
            return is_laurent(Num(unwrap_base), eqsys)
        else
            return is_laurent_denom(Num(unwrap_base), eqsys)
        end
    elseif op == /
        unwrap_num, unwrap_denom = unwrap.(args)
        return is_laurent(Num(unwrap_denom), eqsys) && is_laurent_denom(Num(unwrap_num), eqsys)
    end

    return false
end

function is_laurent_denom(expr::Num, eqsys::EquationSystem)
    key = unwrap(expr)

    val = get(eqsys.cache.is_laurent_denom_cache, key, nothing)
    ! isnothing(val) && return val

    val = _is_laurent_denom_uncached(expr, eqsys)
    eqsys.cache.is_laurent_denom_cache[key] = val
    return val
end

function _is_laurent_uncached(expr::Num, eqsys::EquationSystem)
    unwrap_expr = unwrap(expr)
    if isleaf_expr(unwrap_expr, eqsys)
        return true
    end

    op = operation(unwrap_expr)
    args = arguments(unwrap_expr)

    if (op == +) || (op == *)
        return all(is_laurent(Num(arg), eqsys) for arg in args)
    elseif op == ^
        unwrap_base, unwrap_exp = unwrap.(args)
        if !(unwrap_exp isa Number) || !isnearinteger(unwrap_exp)
            return false
        elseif unwrap_exp < -1.0e-6
            return is_laurent_denom(Num(unwrap_base), eqsys)
        else
            return is_laurent(Num(unwrap_base), eqsys)
        end
    elseif op == /
        unwrap_num, unwrap_denom = unwrap.(args)
        return is_laurent_denom(Num(unwrap_denom), eqsys) && is_laurent(Num(unwrap_num), eqsys)
    end

    return false
end

function is_laurent(expr::Num, eqsys::EquationSystem)
    key = unwrap(expr)

    val = get(eqsys.cache.is_laurent_cache, key, nothing)
    ! isnothing(val) && return val

    val = _is_laurent_uncached(expr, eqsys)
    eqsys.cache.is_laurent_cache[key] = val
    return val
end


# Nonpolynomial candidate detection

function is_atomic_nonpolynomial(expr::Num, eqsys::EquationSystem)
    unwrap_expr = unwrap(expr)
    if isleaf_expr(unwrap_expr, eqsys)
        return false
    end

    op = operation(unwrap_expr)

    if op == ^
        exp = unwrap(arguments(unwrap_expr)[2])
        if exp isa Number && isnearinteger(exp) && 0 < exp
            return false
        end
    end

    return !(
        (op == +) ||
            (op == *) ||
            is_laurent(expr, eqsys)
    )
end

function add_special_children!(terms, expr::Num, eqsys::EquationSystem)
    unwrap_expr = unwrap(expr)
    op = operation(unwrap_expr)
    args = arguments(unwrap_expr)

    return if op == ^
        unwrap_arg1, unwrap_arg2 = unwrap.(args)
        if (unwrap_arg2 isa Number && unwrap_arg2 < 0) ||
                (iscall(unwrap_arg2) && (operation(unwrap_arg2) == *) && isequal(unwrap(arguments(unwrap_arg2)[1]), -1))
            reciprocal_expr = Num(1 / unwrap_expr)
            union!(terms, find_nonpolynomial_terms(reciprocal_expr, eqsys))
        elseif unwrap_arg2 isa Number && (
                (
                    unwrap_arg2 isa Rational &&
                        !(abs(abs(numerator(unwrap_arg2)) - 1) < 1.0e-6)
                ) ||
                    (
                    denominator(rationalize(unwrap_arg2, tol = 1.0e-6)) < 100 &&
                        !(abs(abs(numerator(rationalize(unwrap_arg2, tol = 1.0e-6))) - 1) < 1.0e-6)
                )
            )
            radical_expr = Num(unwrap_arg1^(1 / denominator(rationalize(unwrap_arg2, tol = 1.0e-6))))
            union!(terms, find_nonpolynomial_terms(radical_expr, eqsys))
        end
    elseif op == /
        unwrap_num, unwrap_denom = unwrap.(args)
        reciprocal_denom = Num(1 / unwrap_denom)
        if !(unwrap_num isa Number && abs(abs(unwrap_num) - 1) < 1.0e-6)
            union!(terms, find_nonpolynomial_terms(reciprocal_denom, eqsys))
        end
        if iscall(unwrap_denom) && operation(unwrap_denom) == ^
            unwrap_arg1, _ = unwrap.(arguments(unwrap_denom))
            union!(terms, find_nonpolynomial_terms(Num(1 / unwrap_arg1), eqsys))
        elseif iscall(unwrap_denom) && operation(unwrap_denom) == *
            for arg in arguments(unwrap_denom)
                union!(terms, find_nonpolynomial_terms(Num(1 / arg), eqsys))
                if iscall(arg) && operation(arg) == ^
                    unwrap_arg1, _ = unwrap.(arguments(unwrap(arg)))
                    union!(terms, find_nonpolynomial_terms(Num(1 / unwrap_arg1), eqsys))
                end
            end
        end
    end
end

function find_nonpolynomial_terms_uncached(expr::Num, eqsys::EquationSystem)
    norm_expr = Num(normalize_symbolic_function(expr, eqsys))
    unwrap_expr = unwrap(norm_expr)
    terms = Set{Num}()

    if !isleaf_expr(unwrap_expr, eqsys)
        add_special_children!(terms, norm_expr, eqsys)

        for arg in arguments(unwrap_expr)
            union!(terms, find_nonpolynomial_terms(Num(arg), eqsys))
        end
    end

    if isempty(terms) && is_atomic_nonpolynomial(norm_expr, eqsys)
        push!(terms, norm_expr)
    end

    return terms
end

function find_nonpolynomial_terms(expr::Num, eqsys::EquationSystem)
    norm = unwrap(normalize_symbolic_function(expr, eqsys))

    if haskey(eqsys.cache.find_nonpoly_cache, norm)
        return eqsys.cache.find_nonpoly_cache[norm]
    end

    val = find_nonpolynomial_terms_uncached(Num(norm), eqsys)
    eqsys.cache.find_nonpoly_cache[norm] = val
    return val
end


# Rewriting expressions using already chosen substitutions

function substitute_known_subexpressions(eqsys::EquationSystem, expr::Num)
    norm_expr = Num(normalize_symbolic_function(expr, eqsys))
    key = (unwrap(norm_expr), eqsys.cache.version)

    if haskey(eqsys.cache.substitute_known_cache, key)
        return eqsys.cache.substitute_known_cache[key]
    end

    val = repeated_substitute(norm_expr, rhs_to_lhs_dict_cached(eqsys))
    eqsys.cache.substitute_known_cache[key] = val
    return val
end


# Exponent substitution helpers

function rebuild_with_exponent_substitutions(eqsys::EquationSystem, unwrap_expr)
    rewritten_args = [
        exponent_substitutions(eqsys, arg)
            for arg in arguments(unwrap_expr)
    ]

    return unwrap(normalize_symbolic_function(operation(unwrap_expr)(rewritten_args...), eqsys))
end

function power_substitution_index(eqsys::EquationSystem, unwrap_base, unwrap_exp)
    norm_base = unwrap(normalize_symbolic_function(unwrap_base, eqsys))

    idx = get(eqsys.cache.sub_rhs_index, norm_base, nothing)
    ! isnothing(idx) && return idx

    idx = get(eqsys.cache.sub_power_index, (norm_base, 1), nothing)
    ! isnothing(idx) && return idx

    for ((rhs_base, rhs_exp), idx) in eqsys.cache.sub_power_index
        if equalmag_normalized(rhs_base, norm_base) &&
                isnearinteger(unwrap_exp / rhs_exp)
            return idx
        end
    end

    return nothing
end

function reciprocal_substitution_index(eqsys::EquationSystem, unwrap_denom)
    norm_target = unwrap(normalize_symbolic_function(1 / unwrap_denom, eqsys))

    idx = get(eqsys.cache.sub_rhs_index, norm_target, nothing)
    ! isnothing(idx) && return idx

    for (rhs, idx) in eqsys.cache.sub_rhs_index
        if equalmag_normalized(rhs, norm_target)
            return idx
        end
    end

    return nothing
end

function rewrite_power_case(eqsys::EquationSystem, unwrap_base, unwrap_exp)
    unwrap_sub_index = power_substitution_index(eqsys, unwrap_base, unwrap_exp)

    if isnothing(unwrap_sub_index)
        return nothing
    end

    unwrap_sub = eqsys.substitution_equations[unwrap_sub_index]

    unwrap_base_norm = normalize_symbolic_function(unwrap_base, eqsys)

    rhs = unwrap(unwrap_sub.rhs)

    if equalmag_normalized(rhs, unwrap_base_norm)
        if !isequal(unwrap(unwrap_sub.rhs), unwrap_base_norm)
            return (-unwrap_sub.lhs)^unwrap_exp
        end

        return unwrap_sub.lhs^unwrap_exp
    else
        if !(
                iscall(rhs) &&
                    (operation(rhs) == ^) &&
                    isequal(
                    unwrap_base_norm,
                    unwrap(arguments(rhs)[1])
                )
            )

            return (-unwrap_sub.lhs)^(
                unwrap_exp / unwrap(arguments(rhs)[2])
            )
        end

        return unwrap_sub.lhs^(
            unwrap_exp / unwrap(arguments(rhs)[2])
        )
    end
end

function rewrite_simple_quotient_case(eqsys::EquationSystem, unwrap_num, unwrap_denom)
    unwrap_sub_index = reciprocal_substitution_index(eqsys, unwrap_denom)

    if isnothing(unwrap_sub_index)
        return nothing
    end

    unwrap_sub = eqsys.substitution_equations[unwrap_sub_index]

    target = unwrap(normalize_symbolic_function(1 / unwrap_denom, eqsys))

    if isequal(unwrap(normalize_symbolic_function(unwrap_sub.rhs, eqsys)), target)
        return unwrap_num * unwrap_sub.lhs
    else
        return -unwrap_num * unwrap_sub.lhs
    end
end

function denominator_power_factor_substitution_index(eqsys::EquationSystem, unwrap_base, unwrap_exp)
    norm_target = unwrap(normalize_symbolic_function(1 / unwrap_base, eqsys))

    idx = get(eqsys.cache.sub_rhs_index, norm_target, nothing)
    ! isnothing(idx) && return idx

    idx = get(eqsys.cache.sub_reciprocal_power_index, (norm_target, 1), nothing)
    ! isnothing(idx) && return idx

    for ((rhs_base, rhs_exp), idx) in eqsys.cache.sub_reciprocal_power_index
        if equalmag_normalized(rhs_base, norm_target) &&
                isnearinteger(unwrap_exp / rhs_exp)
            return idx
        end
    end

    return nothing
end

function rewrite_denominator_power_factor(eqsys::EquationSystem, arg)
    arg = unwrap(arg)

    if isleaf_expr(arg, eqsys) || operation(arg) != ^
        return arg
    end

    unwrap_base, unwrap_exp = unwrap.(arguments(arg))
    unwrap_sub_index =
        denominator_power_factor_substitution_index(eqsys, unwrap_base, unwrap_exp)

    if isnothing(unwrap_sub_index)
        return arg
    end

    unwrap_sub = eqsys.substitution_equations[unwrap_sub_index]

    rhs = unwrap(unwrap_sub.rhs)
    lhs = unwrap(unwrap_sub.lhs)
    target = normalize_symbolic_function(1 / unwrap_base, eqsys)

    return if equalmag_normalized(rhs, target)
        if !isequal(rhs, target)
            return (-lhs)^unwrap_exp
        else
            return (lhs)^unwrap_exp
        end
    else
        if !isequal(rhs, target) &&
                !(
                iscall(rhs) &&
                    (operation(rhs) == ^) &&
                    isequal(
                    target,
                    unwrap(arguments(rhs)[1])
                )
            )

            return (-lhs)^(
                unwrap_exp / unwrap(arguments(rhs)[2])
            )
        else
            return (lhs)^(
                unwrap_exp / unwrap(arguments(rhs)[2])
            )
        end
    end
end

function rewrite_factored_denominator_case(eqsys::EquationSystem, unwrap_num, unwrap_denom)
    if isleaf_expr(unwrap_denom, eqsys)
        return nothing
    end

    if !((operation(unwrap_denom) == ^) || (operation(unwrap_denom) == *))
        return nothing
    end

    if operation(unwrap_denom) == *
        args = unwrap.(arguments(unwrap_denom))
    else
        args = [unwrap_denom]
    end

    denom = 1

    for arg in args
        denom *= rewrite_denominator_power_factor(eqsys, arg)
    end

    return unwrap_num / denom
end

"""
Rewrite powers and quotients using the substitutions already stored in an equation
system.

This function recursively rewrites the arguments of an expression, rebuilds the
expression, and then checks whether powers or reciprocals can be expressed using
previously introduced substitution variables. For example, if a substitution variable
`w_0(t)` represents `sqrt(x(t))`, then later occurrences of `x(t)^(3/2)` may be
rewritten in terms of powers of `w_0(t)`.

This is necessary because Symbolics.jl substitute cannot find these substitutions.
"""
function exponent_substitutions(eqsys::EquationSystem, unwrap_expr)
    unwrap_expr = unwrap(normalize_symbolic_function(unwrap_expr, eqsys))

    if isleaf_expr(unwrap_expr, eqsys)
        return unwrap_expr
    end

    rebuilt_expr = unwrap(rebuild_with_exponent_substitutions(eqsys, unwrap_expr))

    if isleaf_expr(rebuilt_expr, eqsys)
        return rebuilt_expr
    end

    op = operation(rebuilt_expr)
    args = arguments(rebuilt_expr)

    if op == ^
        unwrap_base, unwrap_exp = unwrap.(args)

        result = rewrite_power_case(eqsys, unwrap_base, unwrap_exp)
        if result != nothing
            return result
        end

    elseif op == /
        unwrap_num, unwrap_denom = unwrap.(args)

        result = rewrite_simple_quotient_case(eqsys, unwrap_num, unwrap_denom)
        if result != nothing
            return result
        end

        result = rewrite_factored_denominator_case(eqsys, unwrap_num, unwrap_denom)
        if result != nothing
            return result
        end
    end

    return rebuilt_expr
end

function simplify_with_substitutions(eqsys::EquationSystem, expr::Num)
    substituted_expr = substitute_known_subexpressions(eqsys, expr)
    return Num(exponent_substitutions(eqsys, unwrap(substituted_expr)))
end

function try_polynomialize(eqsys::EquationSystem, expr::Num)
    key = (unwrap(expr), eqsys.cache.version)

    if haskey(eqsys.cache.try_polynomialize_cache, key)
        return eqsys.cache.try_polynomialize_cache[key]
    end

    simexpr = simplify_with_substitutions(eqsys, expr)
    result = (simexpr, is_laurent(simexpr, eqsys))

    eqsys.cache.try_polynomialize_cache[key] = result
    return result
end


# Equation-system mutations

function lie_derivative(eqsys::EquationSystem, expr::Num)
    return expand_derivatives(Differential(eqsys.iv)(expr))
end

function add_variable!(eqsys::EquationSystem, expr::Num)
    derv = repeated_substitute(lie_derivative(eqsys, expr), lhs_to_rhs_dict_cached(eqsys))

    new_var = create_variable!(eqsys.holder, eqsys.iv)

    push!(eqsys.eqns, Differential(eqsys.iv)(new_var) ~ derv)

    inserted_keys = Vector{Tuple{Symbol, Any, Bool, Any}}()

    push!(eqsys.substitution_equations, new_var ~ expr)
    index_substitution!(eqsys, length(eqsys.substitution_equations), inserted_keys)

    push!(
        eqsys.substitution_equations,
        1 / new_var ~ Num(normalize_symbolic_function(1 / expr, eqsys))
    )
    index_substitution!(eqsys, length(eqsys.substitution_equations), inserted_keys)

    push!(eqsys.cache.sub_index_history, inserted_keys)

    invalidate_cache!(eqsys)

    return new_var
end

function remove_variable!(eqsys::EquationSystem, var::Num)
    filter!(x -> !isequal(Num(arguments(unwrap(x.lhs))[1]), var), eqsys.eqns)

    pop!(eqsys.substitution_equations)

    pop!(eqsys.substitution_equations)

    delete_variable!(eqsys.holder, var)

    undo_last_substitution_indices!(eqsys)
    invalidate_cache!(eqsys)

    return nothing
end


# Candidate generation and search

function clean_candidate_list(eqsys::EquationSystem, candidates)
    used_rhs = Set{Any}()

    for eqn in eqsys.substitution_equations
        push!(used_rhs, unwrap(normalize_symbolic_function(eqn.rhs, eqsys)))
    end

    return filter(candidates) do x
        ux = unwrap(x)
        !(ux isa Number && isnan(ux)) &&
            !iszero(x) &&
            !(unwrap(normalize_symbolic_function(x, eqsys)) in used_rhs)
    end
end

function available_substitutions(eqsys::EquationSystem)
    bad_expressions = Set{Num}()

    for eqn in eqsys.eqns
        simexpr, laurent = try_polynomialize(eqsys, Num(eqn.rhs))
        if !laurent
            union!(bad_expressions, find_nonpolynomial_terms(simexpr, eqsys))
        end
    end

    candidates = sort(
        collect(bad_expressions),
        by = x -> (expr_depth_cached(eqsys, unwrap(x)), expr_size_cached(eqsys, x))
    )
    non_empty_or_previously_used = clean_candidate_list(eqsys, candidates)

    return non_empty_or_previously_used
end


# Finalizing system

function laurent_system(eqsys::EquationSystem, simplified_eqns)
    poly_subs = Dict{Any, Any}()
    for eqn in eqsys.substitution_equations
        if ! isleaf_expr(unwrap(eqn.lhs), eqsys) && is_polynomial(Num(eqn.rhs), eqsys.holder.state_variables)
            poly_subs[eqn.rhs] = eqn.lhs
        end
    end
    eqns = Equation[]
    for eqn in simplified_eqns
        expr = repeated_substitute(eqn.rhs, poly_subs)
        push!(eqns, eqn.lhs ~ Num(expr))
    end
    @named sys = System(eqns, eqsys.iv)
    return sys
end

function laurent_system_is_polynomial(eqsys::EquationSystem, simplified_eqns)
    poly_subs = Dict{Any, Any}()
    for eqn in eqsys.substitution_equations
        if ! isleaf_expr(unwrap(eqn.lhs), eqsys) && is_polynomial(Num(eqn.rhs), eqsys.holder.state_variables)
            poly_subs[eqn.rhs] = eqn.lhs
        end
    end
    eqns = Equation[]
    for eqn in simplified_eqns
        expr = repeated_substitute(eqn.rhs, poly_subs)
        if ! is_polynomial(Num(expr), eqsys.holder.state_variables)
            return nothing
        end
        push!(eqns, eqn.lhs ~ Num(expr))
    end
    @named sys = System(eqns, eqsys.iv)
    return sys
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

function laurent_expand(expr, eqsys::EquationSystem)
    function helper(expr, denom, eqsys)
        expr = unwrap(normalize_symbolic_function(expr, eqsys))
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
                tot += Num(helper(unwrap_num, denom * unwrap_denom, eqsys))
            else
                tot += term / denom
            end
        end
        return tot
    end
    return helper(expr, 1, eqsys)
end

function leaf_lhs_equations(eqns, eqsys)
    return [eq for eq in eqns if isleaf_expr(eq.lhs, eqsys)]
end

function laurent_system_to_polynomial_system(eqsys::EquationSystem, simplified_eqns)
    vars = copy(eqsys.holder.state_variables)
    new_var = add_variable!(eqsys, 1 / prod(var for var in vars))
    eqns = Equation[]
    dervs = Dict{Any, Num}()
    for eqn in simplified_eqns
        unwrap_righthandside = unwrap(laurent_expand(remove_negative_powers(eqn.rhs), eqsys))
        terms = [unwrap_righthandside]
        if iscall(unwrap_righthandside) && operation(unwrap_righthandside) == +
            terms = unwrap.(arguments(unwrap_righthandside))
        elseif iscall(unwrap_righthandside) && operation(unwrap_righthandside) == -
            terms = unwrap.([arguments(unwrap_righthandside)[1], -arguments(unwrap_righthandside)[2]])
        end
        tot = 0
        for term in terms
            term = unwrap(normalize_symbolic_function(term, eqsys))
            if iscall(term) && operation(term) == /
                unwrap_num, unwrap_denom = unwrap.(arguments(term))
                tup = Tuple(Symbolics.degree(unwrap_denom, var) for var in vars)
                deg = maximum(tup)
                term = Num(new_var^deg * prod(vars[i]^(deg - tup[i]) for i in 1:length(vars)) * unwrap_num)
            end
            tot += term
        end
        push!(eqns, eqn.lhs ~ Num(tot))
        dervs[Num(arguments(eqn.lhs)[1])] = Num(tot)
    end
    diff_new_var = 0
    for var in vars
        diff_new_var += -new_var^2 * prod(other for other in vars if ! isequal(var, other)) * dervs[var]
    end
    push!(eqns, Differential(eqsys.iv)(new_var) ~ expand(diff_new_var))
    solution_substitution_equations = leaf_lhs_equations(eqsys.substitution_equations, eqsys)
    remove_variable!(eqsys, new_var)
    @named sys = System(eqns, eqsys.iv)
    return sys, solution_substitution_equations
end


# Branch and Bound Algorithm

function polynomialize_helper(eqsys::EquationSystem, maxdepth, maxnum, ct, laurent)
    if ct % 100 == 0
        trim_cache!(eqsys)
    end

    subs = available_substitutions(eqsys)
    if isempty(subs)
        simplified_eqns = Equation[]
        for eqn in eqsys.eqns
            simexpr, _ = try_polynomialize(eqsys, Num(eqn.rhs))
            push!(simplified_eqns, eqn.lhs ~ Num(simexpr))
        end
        if laurent
            polysys = laurent_system(eqsys, simplified_eqns)
            return [polysys, length(eqsys.substitution_equations) // 2, ct, leaf_lhs_equations(eqsys.substitution_equations, eqsys)]
        else
            polysys = laurent_system_is_polynomial(eqsys, simplified_eqns)
            if ! isnothing(polysys)
                return [polysys, length(eqsys.substitution_equations) // 2, ct, leaf_lhs_equations(eqsys.substitution_equations, eqsys)]
            elseif isnothing(polysys) && length(eqsys.substitution_equations) // 2 <= maxdepth - 2
                polysys, solution_substitution_equations = laurent_system_to_polynomial_system(eqsys, simplified_eqns)
                return [polysys, length(eqsys.substitution_equations) // 2 + 1, ct, solution_substitution_equations]
            else
                return [nothing, nothing, ct, nothing]
            end
        end
    elseif length(eqsys.substitution_equations) // 2 >= maxdepth - 1
        return [nothing, nothing, ct, nothing]
    end
    final_result = [nothing, nothing, ct, nothing]
    for sub in subs
        new_var = add_variable!(eqsys, Num(normalize_symbolic_function(sub, eqsys; absolute = true)))
        result, num_subs, ct, solution_subs = polynomialize_helper(eqsys, maxdepth, maxnum, ct + 1, laurent)
        final_result[3] = ct
        remove_variable!(eqsys, new_var)
        if result != nothing
            final_result = [result, num_subs, ct, solution_subs]
            maxdepth = num_subs
        end
        if ct >= maxnum
            return final_result
        end
    end
    return final_result
end

function normalize_equations!(eqsys::EquationSystem)
    for i in eachindex(eqsys.eqns)
        eqn = eqsys.eqns[i]
        eqsys.eqns[i] = eqn.lhs ~ Num(normalize_symbolic_function(eqn.rhs, eqsys))
    end
    return eqsys
end


"""
    polynomialize(sys::System; maxdepth = 10, maxnum = 10000,
        laurent = false, new_var_base_name = "w_", start_new_vars_with = 0)

Transform a nonpolynomial autonomous ODE system into a polynomial ODE system by
introducing new state variables for selected subexpressions.

The algorithm searches for nonpolynomial subexpressions in the right-hand sides
of the system. When it chooses a substitution `w_j(t) = φ_j(x(t))`, it adds
`w_j` as a new state variable and computes its differential equation by taking
the Lie derivative of `φ_j` along the current system. The derivative is then
simplified using the substitutions that have already been introduced.

The search continues until the enlarged system is polynomial or until the search
limits are reached. If `laurent` is `false` and the system has only been reduced
to Laurent-polynomial form, the algorithm may introduce one additional
reciprocal-product variable to clear negative powers.

# Arguments

- `sys::System`: the ModelingToolkit system to polynomialize.

# Keywords

- `maxdepth`: maximum allowed number of substitution variables.
- `maxnum`: maximum number of candidate branches to explore.
- `laurent`: whether to allow the returned system to remain Laurent-polynomial.
- `new_var_base_name`: base name used for introduced variables.
- `start_new_vars_with`: first index used for introduced variables.

# Returns

- `polysys`: a polynomial ModelingToolkit system if the search succeeds;
  otherwise `nothing`.
- `substitutions`: equations defining the introduced substitution variables if
  the search succeeds; otherwise `nothing`.

# Example

```julia
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

@variables x(t) y(t)

eqs = [
    D(x) ~ sqrt(x) + 1 / y + log(x + y),
    D(y) ~ exp(x) / (1 + y^2),
]

@mtkcompile sys = System(eqs, t)

polysys, substitutions = polynomialize(sys)
equations(polysys)
```
"""
function polynomialize(sys::System; maxdepth = 10, maxnum = 10000, laurent = false, new_var_base_name = "w_", start_new_vars_with = 0)
    eqsys = EquationSystem(sys, new_var_base_name, start_new_vars_with)
    normalize_equations!(eqsys)
    res = polynomialize_helper(eqsys, maxdepth, maxnum, 0, laurent)
    return res[1], res[4]
end
