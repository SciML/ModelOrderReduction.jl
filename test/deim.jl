using Test, ModelOrderReduction
using ModelingToolkit, MethodOfLines, OrdinaryDiffEq
using DiffEqBase: BrownFullBasicInit
using LinearAlgebra: norm
using SciMLBase: successful_retcode, symbolic_discretize

# construct an ModelingToolkit.ODESystem with non-empty field substitutions
@variables x t v(..) w(..)
Dx = Differential(x)
Dxx = Dx^2
Dt = Differential(t)
const L = 1.0
const ε = 0.015
const b = 0.5
const γ = 2.0
const c = 0.05
f(v) = v * (v - 0.1) * (1.0 - v)
i₀(t) = 50000.0t^3 * exp(-15.0t)
eqs = [
    ε * Dt(v(x, t)) ~ ε^2 * Dxx(v(x, t)) + f(v(x, t)) - w(x, t) + c,
    Dt(w(x, t)) ~ b * v(x, t) - γ * w(x, t) + c,
]
bcs = [
    v(x, 0.0) ~ 0.0,
    w(x, 0) ~ 0.0,
    Dx(v(0, t)) ~ -i₀(t),
    Dx(v(L, t)) ~ 0.0,
]
domains = [
    x ∈ (0.0, L),
    t ∈ (0.0, 14.0),
]
ivs = [x, t]
dvs = [v(x, t), w(x, t)]
pde_sys = PDESystem(eqs, bcs, domains, ivs, dvs; name = :FN)

N = 5 # (minimum number of) equidistant discretization intervals
dx = (L - 0.0) / N
dxs = [x => dx]
order = 2
discretization = MOLFiniteDifference(dxs, t; approx_order = order)
ode_sys, tspan = symbolic_discretize(pde_sys, discretization)
full_prob = discretize(pde_sys, discretization; fallback = false)
@test full_prob isa DAEProblem
sol = solve(full_prob; saveat = 1.0)
@test successful_retcode(sol)

snapshot = Array(sol.original_sol)
pod_dim = 3
deim_sys = @test_nowarn deim(ode_sys, snapshot, pod_dim)

# check the number of dependent variables in the new system
@test length(ModelingToolkit.get_unknowns(deim_sys)) == pod_dim

function is_symbolic_array(expression)
    value = ModelingToolkit.Symbolics.unwrap(expression)
    return value isa AbstractArray ||
        ModelingToolkit.SymbolicUtils.symtype(value) <: AbstractArray
end

reduced_equations = ModelingToolkit.get_eqs(deim_sys)
@test length(reduced_equations) == 1
@test is_symbolic_array(only(reduced_equations).lhs)
reconstruction_equations = ModelingToolkit.get_observed(deim_sys)
@test length(reconstruction_equations) == length(dvs)
@test all(reconstruction_equations) do equation
    is_symbolic_array(equation.lhs) && is_symbolic_array(equation.rhs)
end
reconstruction_parameters = filter(ModelingToolkit.get_ps(deim_sys)) do parameter
    is_symbolic_array(parameter) &&
        size(ModelingToolkit.Symbolics.wrap(parameter)) == (size(snapshot, 1), pod_dim)
end
@test length(reconstruction_parameters) == 1

deim_prob = DAEProblem(
    deim_sys, nothing, tspan;
    initializealg = BrownFullBasicInit(),
    build_initializeprob = false,
)

deim_sol = solve(deim_prob; saveat = 1.0)
@test successful_retcode(deim_sol)

nₓ = length(sol[x])
nₜ = length(sol[t])

# test solution retrieval
@test size(deim_sol[v(x, t)]) == (nₓ, nₜ)
@test size(deim_sol[w(x, t)]) == (nₓ, nₜ)
full_fields = vcat(sol[v(x, t)], sol[w(x, t)])
reduced_fields = vcat(deim_sol[v(x, t)], deim_sol[w(x, t)])
@test norm(reduced_fields - full_fields) / norm(full_fields) < 0.6

@parameters a = 2.0
@variables z(t)[1:4] energy(t)
z_scalars = ModelingToolkit.Symbolics.value.(
    ModelingToolkit.Symbolics.scalarize(z)
)
@named parameterized_array_system = System(
    [Dt(z) ~ -z - a * (z .^ 3)], t, z_scalars, [a];
    observed = [energy ~ sum(z)],
)
parameterized_snapshot = [
    sin(0.3 * i * j) + cos(0.2 * i * (j + 1)) for i in 1:4, j in 1:8
]
parameterized_reduced = deim(parameterized_array_system, parameterized_snapshot, 2)
parameterized_prob = DAEProblem(
    parameterized_reduced, nothing, (0.0, 1.0);
    initializealg = BrownFullBasicInit(),
    build_initializeprob = false,
)
@test parameterized_prob.ps[a] == 2.0
parameterized_sol = solve(parameterized_prob; saveat = 0.2)
@test successful_retcode(parameterized_sol)
@test length(parameterized_sol[energy]) == 6

function tree_size(expression)
    value = ModelingToolkit.Symbolics.unwrap(expression)
    ModelingToolkit.SymbolicUtils.iscall(value) || return 1
    return 1 + sum(
        tree_size, ModelingToolkit.SymbolicUtils.arguments(value); init = 0
    )
end

function reduced_system_size(intervals)
    local_dx = L / intervals
    local_discretization = MOLFiniteDifference(
        [x => local_dx], t; approx_order = order
    )
    local_sys, _ = symbolic_discretize(pde_sys, local_discretization)
    rows = length(ModelingToolkit.get_unknowns(local_sys))
    local_snapshot = [
        sin(0.13 * i * j) + cos(0.07 * i * (j + 1)) for i in 1:rows, j in 1:8
    ]
    local_reduced = deim(local_sys, local_snapshot, 2)
    local_equations = ModelingToolkit.get_eqs(local_reduced)
    local_observed = ModelingToolkit.get_observed(local_reduced)
    return (
        equations = length(local_equations),
        equation_tree = sum(local_equations; init = 0) do equation
            tree_size(equation.lhs) + tree_size(equation.rhs)
        end,
        observed = length(local_observed),
        observed_tree = sum(local_observed; init = 0) do equation
            tree_size(equation.lhs) + tree_size(equation.rhs)
        end,
        array_parameters = count(
            is_symbolic_array, ModelingToolkit.get_ps(local_reduced)
        ),
    )
end

# Fixed POD/DEIM dimensions produce a grid-independent generated symbolic graph.
@test reduced_system_size(5) == reduced_system_size(20)
