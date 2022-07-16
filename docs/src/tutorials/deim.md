# Discrete Empirical Interpolation Method (DEIM)

This is an introductory example for the use of [`deim`](@ref) to the Pleiades problem.

```@example deim-pleiades
using DiffEqProblemLibrary.ODEProblemLibrary
ODEProblemLibrary.importodeproblems()
@doc prob_ode_pleiades
```

[`prob_ode_pleiades`](https://github.com/SciML/DiffEqProblemLibrary.jl/blob/c2cf35b728856c1d9c07d48dc83daef91b964e2c/src/ode/ode_simple_nonlinear_prob.jl#L225)
is an example problem in 
[DiffEqProblemLibrary.jl](https://github.com/JuliaDiffEq/DiffEqProblemLibrary.jl/tree/master/src/ode).
It includes 28 variables with both polynomial and non-polynomial terms.

As DEIM is based on the Proper Orthogonal Decomposition (POD), we firstly sample some 
snapshots.

```@example deim-pleiades
using DifferentialEquations
sol_full = solve(prob_ode_pleiades)
using Plots
plot(sol_full)
```

The singular values of the sanpshot matrix are

```@example deim-pleiades
snapshots = reduce(hcat, sol_full.u)
using LinearAlgebra
vecs, vals = svd(snapshots)
vals
```

We take the first, say, 9 modes as our POD basis.

```@example deim-pleiades
pod_basis = @view vecs[:, 1:9]
```

Then, we call `modelingtoolkitize` on `prob_ode_pleiades` to obtain a symbolic 
representation and apply DEIM.

```@example deim-pleiades
using ModelingToolkit
pleiades_sys = modelingtoolkitize(prob_ode_pleiades)
using ModelOrderReduction
pleiades_sys_deim = deim(pleiades_sys, pod_basis)
length(states(pleiades_sys_deim))
```

```@example deim-pleiades
length(equations(pleiades_sys_deim))
```

We obtain a reduced model with 9 dependent variables and 9 ODEs.

At last, we simply input the initial conditions and time span from the original full order
problem to construct a new `ODEProblem` and solve it.

```@example deim-pleiades
u0 = Dict(states(pleiades_sys) .=> prob_ode_pleiades.u0)
tspan = prob_ode_pleiades.tspan
prob_deim = ODEProblem(pleiades_sys_deim, u0, tspan)
sol_deim = solve(prob_deim)
nothing
```

The plot for the numerical solution of the reduced model is shown as follows.

```@example deim-pleiades
plot(sol_deim)
```

To calculate the approximate solution the original model, one needs to request it
explicitly.

```@example deim-pleiades
plot(sol_deim, vars = states(pleiades_sys))
```
