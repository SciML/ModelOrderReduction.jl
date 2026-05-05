using ModelingToolkit
using Symbolics
using OrdinaryDiffEq
using Statistics
using LinearAlgebra


include("Polynomialization.jl")
include("Quadratization.jl")
include("GalerkinReduction.jl")

unwrap(x) = Symbolics.value(x)

function substitute_fixedpoint(expr, vals; maxiters = 20)
    expr = unwrap(expr)

    dict = Dict{Any, Any}()
    for (k, v) in vals
        dict[k] = v
        dict[unwrap(k)] = v
    end

    for _ in 1:maxiters
        newexpr = Symbolics.substitute(expr, dict)
        isequal(newexpr, expr) && return newexpr
        expr = newexpr
    end

    return expr
end

function numeric_value(expr, vals)
    expr_sub = substitute_fixedpoint(expr, vals)
    return Float64(Symbolics.value(expr_sub))
end

function variable_lhs(lhs, iv)
    ulhs = unwrap(lhs)

    if !iscall(ulhs)
        return Num(ulhs)
    end

    if length(arguments(ulhs)) == 1 && isequal(arguments(ulhs)[1], unwrap(iv))
        return Num(ulhs)
    end

    return nothing
end

function initial_dict(sys, u0_vec)
    xs = unknowns(sys)

    length(xs) == length(u0_vec) ||
        error("Initial condition has length $(length(u0_vec)), but system has $(length(xs)) unknowns.")

    vals = Dict{Any, Any}()

    for (x, v) in zip(xs, u0_vec)
        vals[x] = Float64(v)
        vals[unwrap(x)] = Float64(v)
    end

    return vals
end

function extend_initial_dict!(vals, substitution_equations, iv)
    for eq in substitution_equations
        lhs_var = variable_lhs(eq.lhs, iv)

        lhs_var === nothing && continue

        rhs_val = numeric_value(eq.rhs, vals)

        vals[lhs_var] = rhs_val
        vals[unwrap(lhs_var)] = rhs_val
    end

    return vals
end

function reduced_coordinate_formulas(sys, V, a_vars)
    xs = unknowns(sys)
    n, r = size(V)

    n == length(xs)
    r == length(a_vars)

    formulas = Equation[]

    for i in 1:r
        rhs = zero(Num)

        for j in 1:n
            rhs += V[j, i] * xs[j]
        end

        push!(formulas, a_vars[i] ~ Symbolics.simplify(rhs))
    end

    return formulas
end

function augmented_initial_pairs(old_sys, new_sys, old_u0, substitution_equations)
    iv = ModelingToolkit.get_iv(new_sys)

    vals = initial_dict(old_sys, old_u0)
    extend_initial_dict!(vals, substitution_equations, iv)

    return [x => numeric_value(x, vals) for x in unknowns(new_sys)]
end

function pairs_to_vector(sys, pairs)
    vals = Dict{Any, Any}()

    for p in pairs
        vals[p.first] = p.second
        vals[unwrap(p.first)] = p.second
    end

    return [numeric_value(x, vals) for x in unknowns(sys)]
end

function pod_basis_from_simulation(
        sys, u0_pairs, tspan, nmodes;
        solver = Tsit5(),
        saveat = 0.05,
        solve_kwargs...
    )

    prob = ODEProblem(sys, u0_pairs, tspan)
    sol = solve(prob, solver; saveat = saveat, solve_kwargs...)

    snapshots = reduce(hcat, sol.u)

    xbar = vec(mean(snapshots; dims = 2))
    centered_snapshots = snapshots .- xbar

    F = svd(Matrix(centered_snapshots))
    V = F.U[:, 1:nmodes]

    return V, xbar, sol
end

function make_reduced_variables(iv, nmodes)
    return [
        Symbolics.scalarize(@variables $(Symbol("a_", i))(iv))[1]
            for i in 1:nmodes
    ]
end

"""
    polynomialize_quadratize_reduce(sys, u0, tspan, nmodes;
        polynomialize_kwargs = (
            maxdepth = 6,
            maxnum = 10_000,
            new_var_base_name = "w_",
            start_new_vars_with = 0,
        ),
        quadratize_kwargs = (
            new_var_base_name = "z_",
            start_with = 0,
            max_depth = Inf,
        ),
        solver = Tsit5(),
        saveat = 0.05,
        rom_name = :poly_quad_galerkin_rom,
        solve_kwargs...
    )

Construct an affine Galerkin reduced-order model from a nonpolynomial ODE system.

This function implements the workflow

  1. polynomialize `sys` by introducing auxiliary variables for nonpolynomial expressions,
  2. quadratize the polynomialized system by introducing monomial variables,
  3. simulate the quadratic augmented system and compute a centered POD basis,
  4. construct the affine Galerkin reduced-order model.

The affine reduced subspace is

    X(t) ≈ xbar + V * a(t),

where `X(t)` is the augmented quadratic state, `xbar` is the snapshot mean, `V` is the
POD basis, and `a(t)` is the reduced state.

The reduced model is built using `galerkin_project_system_affine`, so the returned ROM
has observed equations reconstructing the augmented state variables from the affine subspace.

# Arguments

  - `sys`: A `ModelingToolkit.System` representing the original ODE system.
  - `u0`: Initial condition for `sys`. This should be ordered consistently with
    `unknowns(sys)`.
  - `tspan`: Time span passed to `ODEProblem`, for example `(0.0, 10.0)`.
  - `nmodes`: Number of POD modes used in the reduced model.

# Keyword Arguments

  - `polynomialize_kwargs`: Named tuple of keyword arguments forwarded to `polynomialize`.
    The current implementation calls `polynomialize` with `laurent = true`.
  - `quadratize_kwargs`: Named tuple of keyword arguments forwarded to `quadratize`.
  - `solver`: OrdinaryDiffEq solver used for both the augmented quadratic system and the
    reduced system. Defaults to `Tsit5()`.
  - `saveat`: Snapshot spacing used when simulating the quadratic augmented system and the
    reduced system. Defaults to `0.05`.
  - `rom_name`: Name assigned to the returned reduced `System`.
  - `solve_kwargs...`: Additional keyword arguments forwarded to `solve`.

# Returns

A named tuple with fields:

  - `rom`: Completed affine Galerkin reduced-order `System`.
  - `a_vars`: Reduced state variables.
  - `a_formulas`: Projection formulas for the reduced coordinates.
  - `a0`: Reduced initial condition vector.
  - `a0_pairs`: Pair form of the reduced initial condition.
  - `V`: POD basis matrix.
  - `xbar`: Snapshot mean used in the affine subspace.
  - `polysys`: Completed polynomialized system.
  - `quadsys`: Completed quadratized system.
  - `poly_subs`: Polynomialization substitution equations.
  - `quad_subs`: Quadratization substitution equations.
  - `u0_poly`: Initial condition for the polynomialized system.
  - `u0_poly_pairs`: Pair form of the polynomialized initial condition.
  - `u0_quad`: Initial condition for the quadratized system.
  - `u0_quad_pairs`: Pair form of the quadratized initial condition.
  - `u0_quad_solver_order`: Initial augmented state in the solver's ordering.
  - `sol_quad`: Full augmented quadratic solution used to build the POD basis.
  - `sol_rom`: Reduced-order solution.

# Example

    using ModelingToolkit
    using ModelingToolkit: t_nounits as t, D_nounits as D
    using OrdinaryDiffEq

    @variables x(t) y(t)

    eqs = [
        D(x) ~ -x + y + 0.1*sqrt(x),
        D(y) ~ -2.0*y + 0.2*x^2,
    ]

    @mtkcompile sys = System(eqs, t)

    u0 = [1.0, 0.5]
    tspan = (0.0, 10.0)
    nmodes = 2

    result = polynomialize_quadratize_reduce(sys, u0, tspan, nmodes; saveat = 0.01)

    equations(result.rom)
    result.a0
    result.V

# Notes

The POD basis is computed from centered snapshots of the augmented quadratic system. Thus
the Galerkin reduction is affine rather than linear:

    X(t) ≈ xbar + V * a(t)

To reconstruct an original variable such as `x(t)` from the ROM solution, use the
corresponding row of `V` together with `xbar`.
"""
function polynomialize_quadratize_reduce(
        sys,
        u0,
        tspan,
        nmodes;
        polynomialize_kwargs = (
            maxdepth = 6,
            maxnum = 10_000,
            new_var_base_name = "w_",
            start_new_vars_with = 0,
        ),
        quadratize_kwargs = (
            new_var_base_name = "z_",
            start_with = 0,
            max_depth = Inf,
        ),
        solver = Tsit5(),
        saveat = 0.05,
        rom_name = :poly_quad_galerkin_rom,
        solve_kwargs...
    )
    # 0. Complete the original system before doing anything order-sensitive.

    sys = ModelingToolkit.complete(sys)


    # 1. Polynomialize

    polysys_raw, poly_subs = polynomialize(sys; laurent = true, polynomialize_kwargs...)
    polysys = ModelingToolkit.complete(polysys_raw)

    u0_poly_pairs = augmented_initial_pairs(
        sys,
        polysys,
        u0,
        poly_subs,
    )

    u0_poly = pairs_to_vector(polysys, u0_poly_pairs)


    # 2. Quadratize


    quadsys_raw, quad_subs = quadratize(polysys; quadratize_kwargs...)
    quadsys = ModelingToolkit.complete(quadsys_raw)

    u0_quad_pairs = augmented_initial_pairs(
        polysys,
        quadsys,
        u0_poly,
        quad_subs,
    )

    u0_quad = pairs_to_vector(quadsys, u0_quad_pairs)


    # 3. Simulate quadratic system and build POD basis

    V, xbar, sol_quad = pod_basis_from_simulation(
        quadsys,
        u0_quad_pairs,
        tspan,
        nmodes;
        solver = solver,
        saveat = saveat,
        solve_kwargs...
    )

    u0_quad_solver_order = sol_quad.u[1]


    # 4. Galerkin project

    iv = ModelingToolkit.get_iv(quadsys)
    a_vars = make_reduced_variables(iv, nmodes)

    a_formulas = reduced_coordinate_formulas(quadsys, V, a_vars)

    rom_raw = galerkin_project_system_affine(
        quadsys,
        V,
        xbar,
        a_vars;
        name = rom_name,
    )

    rom = ModelingToolkit.complete(rom_raw)

    # POD basis has orthonormal columns.
    a0 = V' * (u0_quad_solver_order .- xbar)
    a0_pairs = [a_vars[i] => a0[i] for i in eachindex(a_vars)]

    prob_rom = ODEProblem(rom, a0_pairs, tspan)
    sol_rom = solve(prob_rom, solver; saveat = saveat, solve_kwargs...)

    return (
        rom = rom,
        a_vars = a_vars,
        a_formulas = a_formulas,
        a0 = a0,
        a0_pairs = a0_pairs,
        V = V,
        xbar = xbar,
        polysys = polysys,
        quadsys = quadsys,
        poly_subs = poly_subs,
        quad_subs = quad_subs,
        u0_poly = u0_poly,
        u0_poly_pairs = u0_poly_pairs,
        u0_quad = u0_quad,
        u0_quad_pairs = u0_quad_pairs,
        u0_quad_solver_order = u0_quad_solver_order,
        sol_quad = sol_quad,
        sol_rom = sol_rom,
    )
end