# ModelOrderReduction.jl

[![Github Action CI](https://github.com/SciML/ModelOrderReduction.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/SciML/ModelOrderReduction.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/SciML/ModelOrderReduction.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/SciML/ModelOrderReduction.jl/tree/main)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

ModelOrderReduction.jl is a package for automatically reducing the computational complexity
of mathematical models, while keeping expected fidelity within a controlled error bound. 
These methods function a submodel with a projection
where solving the smaller model gives approximation information about the full model. 
MOR.jl uses [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl)
as a system description and automatically transforms equations
to the subform, defining the observables to automatically lazily reconstruct the full
model on-demand in a fast and stable form.

## Example
#### Proper Orthogonal Decomposition and Discrete Empirical Interpolation Method (POD-DEIM) on the FitzHugh-Nagumo system
```julia
using ModelingToolkit, MethodOfLines, DifferentialEquations, ModelOrderReduction

# firstly construct a ModelingToolkit.PDESystem for the FitzHugh-Nagumo model
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
bcs = [v(x, 0.0) ~ 0.0, w(x, 0) ~ 0.0, Dx(v(0, t)) ~ -i₀(t), Dx(v(L, t)) ~ 0.0]
domains = [x ∈ (0.0, L), t ∈ (0.0, 14.0)]
ivs = [x, t]
dvs = [v(x, t), w(x, t)]
pde_sys = PDESystem(eqs, bcs, domains, ivs, dvs; name = Symbol("FitzHugh-Nagumo"))

# transfer to a ModelingToolkit.ODESystem by automated discretization via MethodOfLines
N = 15 # equidistant discretization intervals
dx = (L - 0.0) / N
dxs = [x => dx]
discretization = MOLFiniteDifference(dxs, t)
ode_sys, tspan = symbolic_discretize(pde_sys, discretization)
simp_sys = structural_simplify(ode_sys)
ode_prob = ODEProblem(simp_sys, nothing, tspan)

# solve the full-order model to get snapshots
sol = solve(ode_prob, Tsit5())
snapshot_simpsys = Array(sol.original_sol)

# set POD and DEIM dimensions
# apply POD-DEIM to obtain the reduced-order model
pod_dim = deim_dim = 5
deim_sys = deim(simp_sys, snapshot_simpsys, pod_dim; deim_dim = deim_dim)
deim_prob = ODEProblem(deim_sys, nothing, tspan)
deim_sol = solve(deim_prob, Tsit5())

# retrieve the approximate solution of the original full-order model
sol_deim_x = deim_sol[x]
sol_deim_v = deim_sol[v(x, t)]
sol_deim_w = deim_sol[w(x, t)]
```

The following figure shows the comparison of the solutions of the 32-dimension full-order model and the POD5-DEIM5 reduced-order model.

![comparison](https://user-images.githubusercontent.com/45696147/195765614-df9092a2-4fca-4602-bb15-81e65b2b572e.svg)
