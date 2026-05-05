using Base.Iterators: product
using Markdown
using ModelingToolkit
using SymbolicUtils
using Symbolics


# Basic utilities

const ExpTuple = Tuple{Vararg{Int}}

const ONE_NUM = Num(1)

function make_var_from_index(base_name::String, index::Int, iv::Num)
    return Symbolics.scalarize(@variables $(Symbol(base_name, index))(iv))[1]
end

tuple_add(a::ExpTuple, b::ExpTuple) =
    ntuple(i -> a[i] + b[i], length(a))

tuple_sub(a::ExpTuple, b::ExpTuple) =
    ntuple(i -> a[i] - b[i], length(a))

tuple_halffloor(a::ExpTuple) =
    ntuple(i -> fld(a[i], 2), length(a))

is_zero_tuple(a::ExpTuple) =
    all(iszero, a)

unwrap(x) = Symbolics.value(x)

isnearinteger(x; tol = 1.0e-6) = x isa Number && abs(x - round(x)) <= tol

function lhs_to_rhs_dict(eqns)
    return Dict([eq.lhs => eq.rhs for eq in eqns])
end

function expression_in_list(expr, list)
    return any(isequal(expr, elem) for elem in list)
end

function term_to_tuple(term, vars)
    return Tuple(Symbolics.degree(term, var) for var in vars)
end

function tuple_to_term(tuple, vars)
    return prod(vars[i]^tuple[i] for i in 1:length(vars))
end

function num_factors(tuple)
    return prod((abs(i) + 1) for i in tuple)
end

function exponent_range(ai::Integer)
    return ai >= 0 ? (0:ai) : (ai:0)
end

function rhs_terms_from_system(eqns)
    vars = [Num(arguments(eqn.lhs)[1]) for eqn in eqns]
    return [
        expression_to_list(eqn.rhs, vars)
            for eqn in eqns
    ]
end

function derivative_terms_tuple(a::Tuple, rhs_terms)
    out = Set{typeof(a)}()
    n = length(a)

    for i in 1:n
        ai = a[i]
        ai == 0 && continue

        e_i = ntuple(j -> j == i ? 1 : 0, n)

        for b in rhs_terms[i]
            push!(out, tuple_add(tuple_sub(a, e_i), b))
        end
    end

    return collect(out)
end

function expression_to_list(expr, vars)
    rhs = unwrap(expand(expr))

    if iscall(rhs) && operation(rhs) == +
        return [term_to_tuple(arg, vars) for arg in arguments(rhs)]
    elseif iscall(rhs) && operation(rhs) == -
        a, b = arguments(rhs)
        return [term_to_tuple(a, vars), term_to_tuple(-b, vars)]
    else
        return [term_to_tuple(rhs, vars)]
    end
end

laurent_degree(a::Tuple) = sum(abs, a)

same_orthant_between_zero_and(a::Tuple, b::Tuple) =
    all(
    (ai == 0 && bi == 0) ||
        (ai > 0 && 0 <= bi <= ai) ||
        (ai < 0 && ai <= bi <= 0)
        for (ai, bi) in zip(a, b)
)

function canonical_pair(u::Tuple, v::Tuple)
    return u <= v ? (u, v) : (v, u)
end

function tuple_allowed(a, exponent_signs)
    return all(
        begin
                sign = exponent_signs[i]
                sign == 0 || (sign == 1 && a[i] >= 0) || (sign == -1 && a[i] <= 0)
            end
            for i in eachindex(a)
    )
end

# Equation cleanup

isleaf_expr(expr, vars::Vector{Num}) = !iscall(expr) || any(isequal(expr, var) for var in vars)

"""
    normalize_symbolic_function(expr, vars)

Normalize a symbolic expression by recursively rewriting quotients, powers,
products, sums, and differences into a canonical quotient form.

The return value is equivalent to `expr`, but signs and denominators are pushed
toward the top level of the expression tree.

This normalization is used before Laurent expansion so that expressions with
nested divisions or negative signs can be converted more reliably into monomial
terms.
"""
function normalize_symbolic_function(expr, vars)
    function helper(unwrap_expr)
        if unwrap_expr isa Number && isless(unwrap_expr, 0)
            return (-unwrap_expr, 1, -1)
        elseif isleaf_expr(unwrap_expr, vars)
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
            norm_exp = unwrap(normalize_symbolic_function(exponent, vars))
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
                push!(norm_args, normalize_symbolic_function(arg, vars))
            end
            return (operation(unwrap_expr)(norm_args...), 1, 1)
        end
    end
    tup = Num.(helper(unwrap(expr)))
    return tup[3] * tup[1] / tup[2]
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

function laurent_expand(expr, vars)
    function helper(expr, denom, vars)
        expr = unwrap(normalize_symbolic_function(expr, vars))
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
                tot += Num(helper(unwrap_num, denom * unwrap_denom, vars))
            else
                tot += term / denom
            end
        end
        return tot
    end
    return helper(expr, 1, vars)
end

# State / bookkeeping objects

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

struct PolynomialSystem
    orig_variables::Vector{Num}
    orig_eqns::Vector{Equation}
    rhs_terms::Vector{Vector{ExpTuple}}
    substitution_terms::Dict{ExpTuple, Num}
    square_substitutions::Dict{ExpTuple, Tuple{Vararg{Num}}}
    nonsquares::Set{ExpTuple}
    equations_including::Dict{ExpTuple, Vector{Num}}
    holder::VariablesHolder
    iv::Num
    exponent_signs::Vector{Int}
end

"""
    PolynomialSystem(sys::System; new_var_base_name = "z_", start_with = 0)

Constructs the internal monomial representation used by the quadratization
algorithm.

The constructor extracts the state variables and equations from `sys`, expands
the right-hand sides into Laurent monomial terms, and initializes the search
state. The dictionary `substitution_terms` initially contains the constant
monomial and all degree-one monomials corresponding to the original variables.
Every monomial appearing in a right-hand side is recorded in
`equations_including`; monomials that are not yet expressible as products of two
known substitution terms are stored in `nonsquares`.

# Fields

- `orig_variables`: original state variables of the input system.
- `orig_eqns`: original equations after Laurent expansion of the right-hand sides.
- `rhs_terms`: monomial exponent tuples appearing in each original right-hand side.
- `substitution_terms`: map from exponent tuples to their representing variables.
- `square_substitutions`: cached quadratic factorizations of exponent tuples.
- `nonsquares`: monomial exponent tuples not currently known to be quadratic.
- `equations_including`: map from exponent tuples to variables whose equations
  contain those tuples.
- `holder`: state used to create and delete substitution variables during search.
- `iv`: independent variable of the ODE system.
- `exponent_signs`: sign restrictions observed for each original variable exponent.
"""
function PolynomialSystem(sys::System; new_var_base_name = "z_", start_with = 0)
    orig_eqns = deepcopy(equations(sys))
    variables = [Num(arguments(eqn.lhs)[1]) for eqn in orig_eqns]
    for i in 1:length(orig_eqns)
        orig_eqns[i] = orig_eqns[i].lhs ~ laurent_expand(orig_eqns[i].rhs, variables)
    end
    holder = VariablesHolder(variables, new_var_base_name, start_with)
    nonsquares = Set{ExpTuple}()
    equations_including = Dict{ExpTuple, Vector{Num}}()
    square_substitutions = Dict{ExpTuple, Tuple{Vararg{Num}}}()
    exponent_signs = [2 for var in variables]
    for eqn in orig_eqns
        eq_terms = expression_to_list(expand(eqn.rhs), variables)
        for tuple in eq_terms
            for i in 1:length(tuple)
                if exponent_signs[i] == 2 && tuple[i] != 0
                    exponent_signs[i] = Integer(tuple[i] / abs(tuple[i]))
                elseif exponent_signs[i] == 1 && tuple[i] < 0
                    exponent_signs[i] = 0
                elseif exponent_signs[i] == -1 && tuple[i] > 0
                    exponent_signs[i] = 0
                end
            end
            if any(elem < 0 for elem in tuple)
                push!(nonsquares, tuple)
            elseif sum(tuple) > 2
                push!(nonsquares, tuple)
            end
            add_to_equations_including!(equations_including, tuple, Num(arguments(eqn.lhs)[1]))
        end
    end
    for i in 1:length(exponent_signs)
        if exponent_signs[i] == 2
            exponent_signs[i] = 1
        end
    end
    substitution_terms = Dict([Tuple(0 for j in 1:length(variables)) => Num(1)])
    for i in 1:length(variables)
        substitution_terms[Tuple(i == j ? 1 : 0 for j in 1:length(variables))] = variables[i]
    end
    rhs_terms = rhs_terms_from_system(orig_eqns)
    return PolynomialSystem(deepcopy(variables), deepcopy(orig_eqns), rhs_terms, substitution_terms, square_substitutions, nonsquares, equations_including, holder, ModelingToolkit.get_iv(sys), exponent_signs)
end

function add_to_equations_including!(equations_including, tuple, var)
    return if haskey(equations_including, tuple)
        push!(equations_including[tuple], var)
    else
        equations_including[tuple] = [var]
    end
end

function remove_from_equations_including!(equations_including, var)
    to_delete = Set{ExpTuple}()
    for (tuple, eqs) in equations_including
        filter!(x -> ! isequal(x, var), eqs)
        if length(eqs) == 0
            push!(to_delete, tuple)
        end
    end
    for tuple in to_delete
        delete!(equations_including, tuple)
    end
    return
end

# PolynomialSystem bookkeeping utilities

function large_factors(a, polysys::PolynomialSystem)
    ranges = (exponent_range(ai) for ai in a)
    result = []

    seen = Set{Tuple{typeof(a), typeof(a)}}()

    for x in product(ranges...)
        u = Tuple(x)
        v = tuple_sub(a, u)

        if all(iszero, u) || all(iszero, v)
            continue
        end

        p = canonical_pair(u, v)
        if p in seen
            continue
        end
        push!(seen, p)

        factorization = []
        if !haskey(polysys.substitution_terms, u)
            push!(factorization, u)
        end
        if u != v && !haskey(polysys.substitution_terms, v)
            push!(factorization, v)
        end

        if !isempty(factorization)
            push!(result, factorization)
        end
    end

    if !haskey(polysys.substitution_terms, a) && !all(iszero, a)
        push!(result, [a])
    end

    return result
end

function is_known_term(polysys::PolynomialSystem, a::Tuple)
    return haskey(polysys.substitution_terms, a)
end

function quadratic_factorization(polysys::PolynomialSystem, m::ExpTuple)
    var = get(polysys.substitution_terms, m, nothing)
    if var !== nothing
        return (var, Num(1))
    end

    fac = get(polysys.square_substitutions, m, nothing)
    if fac !== nothing
        return fac
    end

    for (u, uvar) in polysys.substitution_terms
        v = tuple_sub(m, u)
        if haskey(polysys.substitution_terms, v)
            return (uvar, polysys.substitution_terms[v])
        end
    end

    return nothing
end

function is_quadratic_known(polysys::PolynomialSystem, m::Tuple)
    return quadratic_factorization(polysys, m) !== nothing
end


# Equation-system mutations

function lie_derivative(expr::Num, polysys::PolynomialSystem)
    return expand_derivatives(Differential(polysys.iv)(expr))
end

function is_square!(polysys::PolynomialSystem, tuple::ExpTuple)
    fac = quadratic_factorization(polysys, tuple)

    if fac === nothing
        return false
    end

    f1, f2 = fac

    if !haskey(polysys.substitution_terms, tuple)
        polysys.square_substitutions[tuple] = (f1, f2)
    end

    return true
end

function add_variable!(polysys::PolynomialSystem, tuple::ExpTuple)
    new_var = create_variable!(polysys.holder, polysys.iv)
    polysys.substitution_terms[tuple] = new_var

    added_terms = derivative_terms_tuple(tuple, polysys.rhs_terms)
    term_remove = Set{ExpTuple}()
    for term in polysys.nonsquares
        comp = tuple_sub(term, tuple)
        if haskey(polysys.substitution_terms, comp)
            polysys.square_substitutions[term] = (polysys.substitution_terms[comp], new_var)
            push!(term_remove, term)
        end
    end
    filter!(x -> !(x in term_remove), polysys.nonsquares)

    for term in added_terms
        add_to_equations_including!(polysys.equations_including, term, new_var)
        if !is_square!(polysys, term)
            push!(polysys.nonsquares, term)
        end
    end
    return new_var
end

function remove_variable!(polysys::PolynomialSystem, tuple::ExpTuple)
    varname = polysys.substitution_terms[tuple]

    delete!(polysys.substitution_terms, tuple)
    remove_from_equations_including!(polysys.equations_including, varname)

    to_delete = ExpTuple[]
    to_recompute = ExpTuple[]

    for (tup, factors) in polysys.square_substitutions
        if !haskey(polysys.equations_including, tup)
            push!(to_delete, tup)
        elseif expression_in_list(varname, factors)
            push!(to_recompute, tup)
        end
    end

    for tup in to_delete
        delete!(polysys.square_substitutions, tup)
    end

    for tup in to_recompute
        delete!(polysys.square_substitutions, tup)
        if !is_square!(polysys, tup)
            push!(polysys.nonsquares, tup)
        end
    end

    tup_remove = ExpTuple[]
    for tup in polysys.nonsquares
        if !haskey(polysys.equations_including, tup)
            push!(tup_remove, tup)
        end
    end
    filter!(x -> !(x in tup_remove), polysys.nonsquares)

    return delete_variable!(polysys.holder, varname)
end

# Upper Bound and Pruning

function upper_bound(polysys::PolynomialSystem)
    return prod(
        begin
                u = max(0, maximum(Symbolics.degree(eqn.rhs, var) for eqn in polysys.orig_eqns))
                l = min(0, minimum(Symbolics.degree(eqn.rhs, var) for eqn in polysys.orig_eqns))
                l == 0 ? (u + 1) : (u - l + 2)
            end
            for var in polysys.orig_variables
    )
end

"""
    quadratic_upper_bound_pruning(polysys::PolynomialSystem)

Return a lower bound on the number of known monomial terms required to finish
the current branch.

For each unresolved monomial in `polysys.nonsquares`, this routine computes the
possible complementary factors relative to the currently known substitutions.
It then estimates how many additional substitution terms are needed before every
remaining nonsquare monomial can become quadratic.

The returned value is a lower bound on the final size of
`polysys.substitution_terms` along the current branch. Therefore, if this value
is at least the current incumbent bound, the branch may be pruned.
"""
function quadratic_upper_bound_pruning(polysys::PolynomialSystem)
    d = []
    for m in polysys.nonsquares
        for (v, _) in polysys.substitution_terms
            diff = tuple_sub(m, v)
            if tuple_allowed(diff, polysys.exponent_signs)
                push!(d, diff)
            end
        end
    end
    sort!(d)
    mults = []
    i = 1
    while i <= length(d)
        start = d[i]
        num = 0
        while i <= length(d) && d[i] == start
            num += 1
            i += 1
        end
        push!(mults, num)
    end
    sort!(mults, rev = true)
    k = 0
    tot = 0
    index = 0
    while length(polysys.nonsquares) > tot
        k += 1
        index += 1
        tot += k + ((index <= length(mults)) ? mults[index] : 0)
    end
    return k + length(polysys.substitution_terms)
end

function C(n, m)
    table = [
        0 1 0 0 0 0 0 0
        1 2 2 0 0 0 0 0
        3 3 4 4 0 0 0 0
        4 5 5 6 6 0 0 0
        6 6 7 7 8 8 0 0
        7 8 9 9 9 10 10 0
        9 10 11 12 12 12 12 12
    ]
    if n <= 7 && m + 1 <= 8
        return table[n, m + 1]
    end
    return n / 2 * (1 + sqrt(4 * n - 3)) + m
end

"""
    squarefree_graph_pruning(polysys::PolynomialSystem)

Return a lower bound on the final number of known monomial terms using the
squarefree-graph pruning test.

The routine first extracts a subcollection of unresolved monomials with distinct
pairwise product behavior. It then counts even exponent vectors and combines
this information with a combinatorial bound for the number of substitutions
needed to resolve the remaining nonsquare monomials.

The returned value is a lower bound on the final size of
`polysys.substitution_terms` along the current branch. If this lower bound is at
least the current incumbent bound, the branch cannot improve the best known
quadratization.
"""
function squarefree_graph_pruning(polysys::PolynomialSystem)
    nonsquares = sort(collect(polysys.nonsquares), by = x -> -laurent_degree(x))
    products = Set([])
    distinct_product_nonsquares = []
    for tuple in nonsquares
        new_products = Set([])
        skip = false
        for tuple2 in distinct_product_nonsquares
            if tuple_add(tuple, tuple2) in products
                skip = true
                break
            else
                push!(new_products, tuple_add(tuple, tuple2))
            end
        end
        if ! skip
            push!(distinct_product_nonsquares, tuple)
            union!(products, new_products)
        end
    end

    d = []
    c = 0
    for m in distinct_product_nonsquares
        if all(a % 2 == 0 for a in m)
            c += 1
        end
        for (v, _) in polysys.substitution_terms
            diff = tuple_sub(m, v)
            if tuple_allowed(diff, polysys.exponent_signs)
                push!(d, diff)
            end
        end
    end
    sort!(d)
    mults = []
    i = 1
    while i <= length(d)
        start = d[i]
        num = 0
        while i <= length(d) && d[i] == start
            num += 1
            i += 1
        end
        push!(mults, num)
    end
    sort!(mults, rev = true)

    k = 1
    tot = (length(mults) > 0) ? mults[1] : 0
    index = 1
    while length(distinct_product_nonsquares) > tot + C(k, c)
        k += 1
        index += 1
        tot += (index <= length(mults)) ? mults[index] : 0
    end
    k -= 1
    return k + length(polysys.substitution_terms)
end


# Finalizing equations and substitutions

function quadratic_replacement(polysys::PolynomialSystem, term)
    tup = term_to_tuple(term, polysys.orig_variables)
    monomial = tuple_to_term(tup, polysys.orig_variables)
    coeff = Num(term / monomial)

    fac = quadratic_factorization(polysys, tup)

    f1, f2 = fac
    return coeff * f1 * f2
end

function is_original_or_constant_tuple(tuple::ExpTuple)
    nonzero_count = 0

    @inbounds for x in tuple
        if x != 0
            x == 1 || return false
            nonzero_count += 1
            nonzero_count > 1 && return false
        end
    end

    return true
end

function substitution_equations(
        polysys::PolynomialSystem;
        include_originals::Bool = false,
        include_constant::Bool = false
    )
    eqs = Equation[]

    for (tuple, var) in polysys.substitution_terms
        if is_zero_tuple(tuple) && !include_constant
            continue
        end

        if is_original_or_constant_tuple(tuple) && !include_originals
            continue
        end

        monomial = tuple_to_term(tuple, polysys.orig_variables)
        push!(eqs, var ~ monomial)
    end

    return eqs
end

function new_system(polysys::PolynomialSystem)
    eqns = Equation[]

    # First add the original equations.
    for eqn in polysys.orig_eqns
        unwrap_righthandside = unwrap(expand(eqn.rhs))
        terms = [unwrap_righthandside]

        if iscall(unwrap_righthandside) && operation(unwrap_righthandside) == +
            terms = unwrap.(arguments(unwrap_righthandside))
        elseif iscall(unwrap_righthandside) && operation(unwrap_righthandside) == -
            terms = unwrap.([arguments(unwrap_righthandside)[1], -arguments(unwrap_righthandside)[2]])
        end

        tot = 0
        for term in terms
            tot += quadratic_replacement(polysys, term)
        end

        push!(eqns, eqn.lhs ~ Num(tot))
    end

    # Then add equations for substitution variables.
    rhs_dict = lhs_to_rhs_dict(polysys.orig_eqns)
    for (tuple, var) in polysys.substitution_terms
        if is_original_or_constant_tuple(tuple)
            continue
        end

        expr = tuple_to_term(tuple, polysys.orig_variables)
        derv = laurent_expand(
            substitute(
                lie_derivative(expr, polysys),
                rhs_dict
            ), polysys.orig_variables
        )

        unwrap_righthandside = unwrap(derv)
        terms = [unwrap_righthandside]

        if iscall(unwrap_righthandside) && operation(unwrap_righthandside) == +
            terms = unwrap.(arguments(unwrap_righthandside))
        elseif iscall(unwrap_righthandside) && operation(unwrap_righthandside) == -
            terms = unwrap.([arguments(unwrap_righthandside)[1], -arguments(unwrap_righthandside)[2]])
        end

        tot = 0
        for term in terms
            tot += quadratic_replacement(polysys, term)
        end

        push!(eqns, Differential(polysys.iv)(var) ~ Num(tot))
    end

    @named sys = System(eqns, polysys.iv)
    return sys, substitution_equations(polysys)
end

function finalize_from_substitutions(
        polysys::PolynomialSystem,
        best_substitutions::Dict{ExpTuple, Num},
    )
    old_substitutions = copy(polysys.substitution_terms)
    old_square_substitutions = copy(polysys.square_substitutions)

    empty!(polysys.substitution_terms)
    for (k, v) in best_substitutions
        polysys.substitution_terms[k] = v
    end

    empty!(polysys.square_substitutions)

    result = new_system(polysys)

    empty!(polysys.substitution_terms)
    for (k, v) in old_substitutions
        polysys.substitution_terms[k] = v
    end

    empty!(polysys.square_substitutions)
    for (k, v) in old_square_substitutions
        polysys.square_substitutions[k] = v
    end

    return result
end

# Branch and bound algorithm

function choose_branch_monomial(polysys::PolynomialSystem)
    best = nothing
    best_factors = nothing

    for m in polysys.nonsquares
        fs = large_factors(m, polysys)
        if best === nothing || length(fs) < length(best_factors)
            best = m
            best_factors = fs
        end
    end

    return best, best_factors
end

function qbee_style_score(polysys::PolynomialSystem)
    return sum([laurent_degree(m) for m in polysys.nonsquares]) +
        length(polysys.orig_variables) * length(polysys.substitution_terms)
end

function add_factorization!(polysys::PolynomialSystem, factorization)
    added = ExpTuple[]

    for factor in factorization
        if !haskey(polysys.substitution_terms, factor)
            add_variable!(polysys, factor)
            push!(added, factor)
        end
    end

    return added
end

function remove_factorization!(polysys::PolynomialSystem, added)
    for factor in reverse(added)
        remove_variable!(polysys, factor)
    end

    return nothing
end

function branch_score_after_adding(polysys::PolynomialSystem, factorization)
    added = ExpTuple[]

    added = add_factorization!(polysys, factorization)
    score = qbee_style_score(polysys)
    remove_factorization!(polysys, added)
    return score
end

function substitution_snapshot(polysys::PolynomialSystem)
    return copy(polysys.substitution_terms)
end

"""
    branch_bound_alg(polysys::PolynomialSystem, N::Int, best_substitutions)

Search for a small monomial quadratization using recursive branch and bound.

The search state is stored in `polysys`. If all monomial terms are already
quadratic in the known substitution variables, the current substitution set is a
candidate solution. Otherwise, the algorithm chooses an unresolved monomial,
branches over candidate factorizations of that monomial, and recursively adds
the missing factors as new substitution variables.

Branches are discarded when their current size or one of the pruning lower
bounds proves that they cannot improve the incumbent bound `N`.

# Returns

A pair `(N, best_substitutions)`, where `N` is the size of the best substitution
dictionary found so far and `best_substitutions` is the corresponding dictionary
from exponent tuples to symbolic variables.
"""
function branch_bound_alg(
        polysys::PolynomialSystem,
        N::Int,
        best_substitutions::Dict{ExpTuple, Num},
    )
    if isempty(polysys.nonsquares)
        candidate = substitution_snapshot(polysys)

        if length(candidate) < N
            return length(candidate), candidate
        else
            return N, best_substitutions
        end
    end

    min_quadratization_size = length(polysys.substitution_terms) + 1

    if length(polysys.substitution_terms) + 1 >= N ||
            quadratic_upper_bound_pruning(polysys) >= N ||
            squarefree_graph_pruning(polysys) >= N
        return N, best_substitutions
    end

    _, factorizations = choose_branch_monomial(polysys)

    sort!(factorizations, by = x -> length(polysys.orig_variables) * length(x) + sum(laurent_degree(tup) for tup in x))

    for factorization in factorizations
        if length(factorization) == 2
            factor1, factor2 = factorization

            add_variable!(polysys, factor1)
            add_variable!(polysys, factor2)

            N, best_substitutions = branch_bound_alg(
                polysys,
                N,
                best_substitutions,
            )

            remove_variable!(polysys, factor2)
            remove_variable!(polysys, factor1)
        else
            factor1, = factorization

            add_variable!(polysys, factor1)

            N, best_substitutions = branch_bound_alg(
                polysys,
                N,
                best_substitutions,
            )

            remove_variable!(polysys, factor1)
        end
    end

    return N, best_substitutions
end

# Greedy quadratization upper bound

function is_known_quadratic_from_set(m, known)
    if m in known
        return true
    end

    for u in known
        v = tuple_sub(m, u)
        if v in known
            return true
        end
    end

    return false
end

function substitution_dict_from_tuples(
        polysys::PolynomialSystem,
        added_order::Vector{ExpTuple},
    )
    result = copy(polysys.substitution_terms)

    next_index = length(polysys.holder.state_variables) +
        polysys.holder.start_with -
        polysys.holder.num_vars_orig

    for tuple in added_order
        if !haskey(result, tuple)
            result[tuple] = make_var_from_index(
                polysys.holder.base_name,
                next_index,
                polysys.iv,
            )
            next_index += 1
        end
    end

    return result
end

function greedy_initial_substitutions(polysys::PolynomialSystem, upperbound = Inf)
    numeric_upperbound = min(upperbound, upper_bound(polysys))

    known = Set{ExpTuple}(keys(polysys.substitution_terms))
    added = Set{ExpTuple}()
    added_order = ExpTuple[]

    function add_known!(w::ExpTuple)
        return if !(w in known)
            push!(known, w)
            push!(added, w)
            push!(added_order, w)
        end
    end

    function ensure_square(m::ExpTuple)
        if length(known) >= numeric_upperbound
            return
        end

        is_known_quadratic_from_set(m, known) && return

        u = tuple_halffloor(m)
        v = tuple_sub(m, u)

        for w in (u, v)
            is_zero_tuple(w) && continue

            if !(w in known)
                add_known!(w)

                for term in derivative_terms_tuple(w, polysys.rhs_terms)
                    ensure_square(term)

                    if length(known) >= numeric_upperbound
                        return
                    end
                end
            end
        end
        return
    end

    for m in polysys.nonsquares
        ensure_square(m)

        if length(known) >= numeric_upperbound
            break
        end
    end

    return substitution_dict_from_tuples(polysys, added_order)
end

"""
    quadratize(sys::System; new_var_base_name = "z_", start_with = 0,
        max_depth = Inf)

Transform an autonomous ODE system into a quadratic lifted system by introducing
monomial substitution variables.

The input system is converted to an internal Laurent-monomial representation.
The algorithm then searches for monomial substitutions of the form

    z_j(t) = x_1(t)^a_{j1} * x_2(t)^a_{j2} * ... * x_n(t)^a_{jn}

such that every right-hand side in the lifted system has degree at most two in
the original and introduced variables.

# Arguments

- `sys::System`: the ModelingToolkit ODE system to quadratize.

# Keywords

- `new_var_base_name`: base name used for introduced variables.
- `start_with`: first index used for introduced variables.
- `max_depth`: upper bound used when constructing the initial greedy solution.

# Returns

- `quadsys`: a ModelingToolkit system with quadratic right-hand sides.
- `substitution_eqs`: equations defining the introduced monomial variables.

# Example

```julia
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

@variables x(t) y(t) z(t)

eqs = [
    D(x) ~ x^3 + y * z,
    D(y) ~ x * y^2,
    D(z) ~ x^2 * z^2,
]

@mtkcompile sys = System(eqs, t)

quadsys, substitution_eqs = quadratize(sys)
```
"""
function quadratize(sys::System; new_var_base_name = "z_", start_with = 0, max_depth = Inf)
    polysys = PolynomialSystem(sys; new_var_base_name, start_with)

    greedy_substitutions = greedy_initial_substitutions(polysys, max_depth)
    N0 = length(greedy_substitutions)

    _, best_substitutions = branch_bound_alg(
        polysys,
        N0,
        greedy_substitutions,
    )

    return finalize_from_substitutions(polysys, best_substitutions)
end
