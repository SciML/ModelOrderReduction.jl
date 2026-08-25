using Test, ModelOrderReduction
using ModelingToolkit, MethodOfLines, OrdinaryDiffEq
using SciMLBase: DAEProblem, discretize, successful_retcode

# Construct a MethodOfLines v1 array-form DAE with algebraic boundary conditions.
@independent_variables x t
@variables v(..) w(..)
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
dae_prob = discretize(pde_sys, discretization; fallback = false)
@test dae_prob isa DAEProblem
@test length(ModelingToolkit.get_eqs(dae_prob.f.sys)) == 4
@test length(ModelingToolkit.get_unknowns(dae_prob.f.sys)) == 12
sol = solve(dae_prob; saveat = 1.0)
@test successful_retcode(sol)

pod_dim = 3
deim_sys = @test_nowarn deim(dae_prob, sol, pod_dim)

# check the number of dependent variables in the new system
@test length(ModelingToolkit.get_unknowns(deim_sys)) == pod_dim
@test isempty(ModelingToolkit.initialization_equations(deim_sys))

deim_prob = ODEProblem(complete(deim_sys), nothing, dae_prob.tspan)

deim_sol = solve(deim_prob, Rodas5P(), saveat = 1.0)
@test successful_retcode(deim_sol)

nₓ = length(sol[x])
nₜ = length(sol[t])

# Test solution retrieval through the MethodOfLines metadata.
@test size(deim_sol[v(x, t)]) == (nₓ, nₜ)
@test size(deim_sol[w(x, t)]) == (nₓ, nₜ)

# Keep the explicit ODESystem entry point covered without scalarizing the PDE example.
@variables z₁(t) z₂(t)
D = Differential(t)
@mtkcompile explicit_sys = System(
    [D(z₁) ~ z₁ - z₁^3, D(z₂) ~ -z₂], t; name = :explicit_deim_test
)
explicit_snapshot = [1.0 0.8 0.6; 0.5 0.4 0.3]
explicit_deim_sys = @test_nowarn deim(explicit_sys, explicit_snapshot, 1)
@test length(ModelingToolkit.get_unknowns(explicit_deim_sys)) == 1
@test isempty(ModelingToolkit.get_guesses(explicit_deim_sys))
@test isempty(ModelingToolkit.initialization_equations(explicit_deim_sys))
@test length(ModelingToolkit.get_initial_conditions(explicit_deim_sys)) == 1
