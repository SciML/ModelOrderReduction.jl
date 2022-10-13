# Discrete Empirical Interpolation Method (DEIM)

This section illustrates how ModelOrderReduction.jl can be used to build a reduced order 
model via Petrov-Galerkin projection using the Proper Orthogonal Decomposition (POD) and 
the Discrete Empirical Interpolation Method (DEIM). As an illustrative example, we consider 
a nonlinear 1-D PDE from the realm of neuroscience.

The FitzHugh-Nagumo system used in neuron modeling is a simplified version of the 
Hodgkin-Huxley model, which describes in a detailed manner activation and deactivation 
dynamics of a spiking neuron. The system is given as follows. For ``x\in [0,L], t\geq 0``,

```math
\begin{aligned}
\varepsilon v_t(x,t)&=\varepsilon^2 v_{xx}(x,t)+f(v(x,t))-w(x,t)+c,\\
w_t(x,t)&=bv(x,t)-\gamma w(x,t)+c,
\end{aligned}
```

with nonlinear function ``f(v)=v(v-0.1)(1-v)``. The initial and boundary conditions are

```math
\begin{aligned}
v(x,0)=0,\quad w(x,0)=0,\quad x\in [0,L],\\
v_x(0,t)=-i_0(t),\quad v_x(L,t)=0,\quad t\geq 0,
\end{aligned}
```

where the parameters are ``L=1``, ``\varepsilon=0.015``, ``b=0.5``, ``\gamma =2``, 
``c=0.05``. The stimulus is ``i_0(t)=50000t^3\exp(-15t)``. The variables ``v`` and ``w`` 
are voltage and recovery of voltage, respectively.

In order to generate a POD-DEIM reduced-order model, we need to work through the following 
steps:

1. Collect data on full-order model trajectories and the nonlinear terms describing its evolution equation along the way.
1. Based on the collected data, use POD to identify a low dimensional linear subspace of the system's state space that allows for embedding the full-order model's trajectories with minimal error.
1. Project the model onto the identified subspace using DEIM to approximate nonlinear terms.

For step 1, we first construct a 
[`ModelingToolkit.PDESystem`](https://mtk.sciml.ai/stable/systems/PDESystem/) 
describing the original FitzHugh-Nagumo model.

```@example deim_FitzHugh_Nagumo
using ModelingToolkit
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
pde_sys = PDESystem(eqs, bcs, domains, ivs, dvs; name = Symbol("FitzHugh-Nagumo"))
nothing # hide
```

Next, we apply finite difference discretization using 
[MethodOfLines.jl](https://github.com/SciML/MethodOfLines.jl).

```@example deim_FitzHugh_Nagumo
using MethodOfLines
N = 15 # equidistant discretization intervals
dx = (L - 0.0) / N
dxs = [x => dx]
order = 2
discretization = MOLFiniteDifference(dxs, t; approx_order = order)
ode_sys, tspan = symbolic_discretize(pde_sys, discretization)
simp_sys = structural_simplify(ode_sys)
ode_prob = ODEProblem(simp_sys, nothing, tspan)
nothing # hide
```

The snapshot trajectories are obtained by solving the full-order system. 

```@example deim_FitzHugh_Nagumo
using DifferentialEquations
sol = solve(ode_prob, Tsit5())
sol_x = sol[x]
nₓ = length(sol_x) # number of discretization points in x
nₜ = length(sol[t]) # number of discretization points in time
nothing # hide
```

Let's see the fast decay of the singular values of the snapshot solutions for ``v``, ``w``,
and the nonlinear snapshots ``f(v)``.

```@example deim_FitzHugh_Nagumo
using LinearAlgebra
snapshot_v = sol[v(x, t)]
snapshot_w = sol[w(x, t)]
snapshot_fv = f.(snapshot_v)
svdval_v = svdvals(snapshot_v)
svdval_w = svdvals(snapshot_w)
svdval_fv = svdvals(snapshot_fv)
using Plots, LaTeXStrings
svd_plt = plot(yscale = :log10, xticks = eachindex(svdval_v), titlefont = 11,
               legendfont = 10, title = "Singular values of the snapshots")
plot!(svd_plt, svdval_v, markershape = :circle, label = L"Singular Val of $v$")
plot!(svd_plt, svdval_w, markershape = :circle, label = L"Singular Val of $w$")
plot!(svd_plt, svdval_fv, markershape = :circle, label = L"Singular Val of $f(v)$")
```

The following figure shows the phase-space diagram of ``v`` and ``w`` at different spatial
points ``x`` from the full-order system.

```@example deim_FitzHugh_Nagumo
full_plt = plot(xlabel = L"v(x,t)", ylabel = L"x", zlabel = L"w(x,t)", xlims = (-0.5, 2.0),
                ylims = (0.0, L), zlims = (0.0, 0.25), legend = false, xflip = true,
                camera = (50, 30), titlefont = 10,
                title = "Phase−Space diagram of full $(nameof(pde_sys)) system")
@views for i in 1:nₓ
    plot!(full_plt, snapshot_v[i, :], _ -> sol_x[i], snapshot_w[i, :])
end
plot!(full_plt)
```

Then, we use POD to construct a linear subspace of dimension, say, 5 for the system's state
space and project the model onto the subspace. DEIM is employed to approximate nonlinear 
terms. This can be done by simply calling [`deim`](@ref).

```@example deim_FitzHugh_Nagumo
using ModelOrderReduction
snapshot_simpsys = Array(sol.original_sol)
pod_dim = deim_dim = 5
deim_sys = deim(simp_sys, snapshot_simpsys, pod_dim)
deim_prob = ODEProblem(deim_sys, nothing, tspan)
deim_sol = solve(deim_prob, Tsit5())
nₜ_deim = length(deim_sol[t])
sol_deim_x = deim_sol[x]
sol_deim_v = deim_sol[v(x, t)]
sol_deim_w = deim_sol[w(x, t)]
nothing # hide
```

And plot the result from the POD-DEIM reduced system.

```@example deim_FitzHugh_Nagumo
deim_plt = plot(xlabel = L"v(x,t)", ylabel = L"x", zlabel = L"w(x,t)", xlims = (-0.5, 2.0),
                ylims = (0.0, L), zlims = (0.0, 0.25), legend = false, xflip = true,
                camera = (50, 30), titlefont = 10,
                title = "Phase−Space diagram of reduced $(nameof(pde_sys)) system")
@views for i in 1:nₓ
    plot!(deim_plt, sol_deim_v[i, :], _ -> sol_deim_x[i], sol_deim_w[i, :])
end
plot!(deim_plt)
```

Finally, we put the two solutions in one figure.
```@example deim_FitzHugh_Nagumo
# create data for plotting unconnected lines
function unconnected(m::AbstractMatrix)
    row, col = size(m)
    data = similar(m, row, col + 1)
    data[:, begin:(end - 1)] .= m
    data[:, end] .= NaN # path separator
    vec(data')
end
function unconnected(v::AbstractVector, nₜ::Integer)
    data = similar(v, nₜ + 1, length(v))
    for (i, vᵢ) in enumerate(v)
        data[begin:(end - 1), i] .= vᵢ
    end
    data[end, :] .= NaN
    vec(data)
end
plt_2 = plot(xlabel = L"v(x,t)", ylabel = L"x", zlabel = L"w(x,t)", xlims = (-0.5, 2.0),
             ylims = (0.0, L), zlims = (0.0, 0.25), xflip = true, camera = (50, 30),
             titlefont = 10, title = "Comparison of full and reduced systems")
plot!(plt_2, unconnected(snapshot_v), unconnected(sol_x, nₜ), unconnected(snapshot_w),
      label = "Full$(length(ode_sys.eqs))")
plot!(plt_2, unconnected(sol_deim_v), unconnected(sol_deim_x, nₜ_deim),
      unconnected(sol_deim_w), label = "POD$(pod_dim)/DEIM$(deim_dim)")
```

As we can see, the reduced-order system captures the limit cycle of the original full-order 
system very well.
