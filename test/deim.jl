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
sol = solve(ode_prob)

snapshot_simpsys = Array(sol)
pod_dim = 3
pod_reducer = POD(snapshot_simpsys, pod_dim)
reduce!(pod_reducer, TSVD())
pod_basis = pod_reducer.rbasis
deim_sys = @test_nowarn deim(simp_sys, pod_basis)

# check the number of dependent variables in the new system
@test length(ModelingToolkit.get_states(deim_sys)) == pod_dim

deim_prob = ODEProblem(deim_sys, nothing, tspan)

# check projection for initial values
@test pod_basis' * ode_prob.u0 ≈ deim_prob.u0

deim_sol = solve(deim_prob)

grid = get_discrete(pde_sys, discretization)
grid_v = grid[v(x, t)]
grid_w = grid[w(x, t)]

# expect no exception when retrieving eliminated variables
@test_nowarn deim_sol[grid_v]
@test_nowarn deim_sol[grid_w]
