using Test, ModelOrderReduction
using ModelingToolkit, MethodOfLines, DifferentialEquations

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
eqs = [ε * Dt(v(x, t)) ~ ε^2 * Dxx(v(x, t)) + f(v(x, t)) - w(x, t) + c,
    Dt(w(x, t)) ~ b * v(x, t) - γ * w(x, t) + c]
bcs = [v(x, 0.0) ~ 0.0,
    w(x, 0) ~ 0.0,
    Dx(v(0, t)) ~ -i₀(t),
    Dx(v(L, t)) ~ 0.0]
domains = [x ∈ (0.0, L),
    t ∈ (0.0, 14.0)]
ivs = [x, t]
dvs = [v(x, t), w(x, t)]
pde_sys = PDESystem(eqs, bcs, domains, ivs, dvs; name = :FN)

N = 5 # (minimum number of) equidistant discretization intervals
dx = (L - 0.0) / N
dxs = [x => dx]
order = 2
discretization = MOLFiniteDifference(dxs, t; approx_order = order)
ode_sys, tspan = symbolic_discretize(pde_sys, discretization)
simp_sys = structural_simplify(ode_sys) # field substitutions is non-empty
ode_prob = ODEProblem(simp_sys, nothing, tspan)
sol = solve(ode_prob, Tsit5(), saveat = 1.0)

snapshot_simpsys = Array(sol.original_sol)
pod_dim = 3
deim_sys = @test_nowarn deim(simp_sys, snapshot_simpsys, pod_dim)

# check the number of dependent variables in the new system
@test length(ModelingToolkit.get_states(deim_sys)) == pod_dim

deim_prob = ODEProblem(deim_sys, nothing, tspan)

deim_sol = solve(deim_prob, Tsit5(), saveat = 1.0)

nₓ = length(sol[x])
nₜ = length(sol[t])

# test solution retrival
@test size(deim_sol[v(x, t)]) == (nₓ, nₜ)
@test size(deim_sol[w(x, t)]) == (nₓ, nₜ)
