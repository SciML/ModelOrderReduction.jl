# ModelOrderReduction.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/ModelOrderReduction/stable/)

[![codecov](https://codecov.io/gh/SciML/ModelOrderReduction.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/SciML/ModelOrderReduction.jl)
[![Build Status](https://github.com/SciML/ModelOrderReduction.jl/workflows/CI/badge.svg)](https://github.com/SciML/ModelOrderReduction.jl/actions?query=workflow%3ACI)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

ModelOrderReduction.jl is a package for automatically reducing the computational complexity
of mathematical models, while keeping expected fidelity within a controlled error bound.
These methods construct a submodel via a projection
where solving the smaller model gives approximate information about the full model.
MOR.jl uses [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/)
as a system description and automatically transforms equations
to the subform, defining the observables to automatically lazily reconstruct the full
model on-demand in a fast and stable form.

## Tutorials and Documentation

For information on using the package,
[see the stable documentation](https://docs.sciml.ai/ModelOrderReduction/stable/). Use the
[in-development documentation](https://docs.sciml.ai/ModelOrderReduction/dev/) for the version of
the documentation, which contains the unreleased features.

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

#### Polynomialization, Quadratization, and Galerkin Reduction on an ODE system
```julia
using ModelOrderReduction
using ModelingToolkit
using OrdinaryDiffEq
using LinearAlgebra
using Symbolics

using ModelingToolkit: t_nounits as t, D_nounits as D

@variables x(t) y(t)

eqs = [
    D(x) ~ -x + y + 0.1 * sqrt(x),
    D(y) ~ -2.0 * y + 0.2 * x^2,
]

@mtkcompile sys = System(eqs, t)
sys = ModelingToolkit.complete(sys)

u0_pairs = [
    x => 0.5,
    y => 1.0,
]

tspan = (0.0, 1.0)
saveat = 0.1
nmodes = 2

# Polynomialization + quadratization
quadsys_raw, quad_subs = polynomialize_and_quadratize(sys)

quadsys = ModelingToolkit.complete(quadsys_raw)

# Compute augmented initial conditions
u0_quad_pairs = compute_augmented_initial_pairs(
    sys,
    quadsys,
    u0_pairs,
    quad_subs,
)

# Solve quadratized system
prob_quad = ODEProblem(quadsys, u0_quad_pairs, tspan)
sol_quad = solve(prob_quad, Tsit5(); saveat = saveat)

snapshots = reduce(hcat, sol_quad.u)

xbar = [sum(snapshots[i, :]) / size(snapshots, 2) for i in axes(snapshots, 1)]
centered_snapshots = snapshots .- reshape(xbar, :, 1)

# POD basis
F = svd(Matrix(centered_snapshots))
V = F.U[:, 1:nmodes]

nquad = length(ModelingToolkit.unknowns(quadsys))

# Initial reduced coordinates
u0_quad_solver_order = sol_quad.u[1]
a0 = V' * (u0_quad_solver_order .- xbar)

iv = ModelingToolkit.get_iv(quadsys)

a_vars = [
    Symbolics.scalarize(@variables $(Symbol("a_", i))(iv))[1]
    for i in 1:nmodes
]

# Galerkin projection
rom_raw = galerkin_project_system_affine(
    quadsys,
    V,
    xbar,
    a_vars,
)

rom = ModelingToolkit.complete(rom_raw)

rom_unknowns = ModelingToolkit.unknowns(rom)

a0_pairs = [a_vars[i] => a0[i] for i in eachindex(a_vars)]

# Solve ROM
prob_rom = ODEProblem(rom, a0_pairs, tspan)
sol_rom = solve(prob_rom, Tsit5(); saveat = saveat)

```
