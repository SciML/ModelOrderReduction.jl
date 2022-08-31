```@example FitzHugh_Nagumo
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

using MethodOfLines
N = 15 # equidistant discretization intervals
dx = (L - 0.0) / N
dxs = [x => dx]
order = 2
discretization = MOLFiniteDifference(dxs, t; approx_order = order)
ode_sys, tspan = symbolic_discretize(pde_sys, discretization)
simp_sys = structural_simplify(ode_sys)
ode_prob = ODEProblem(simp_sys, nothing, tspan)

using DifferentialEquations
sol = solve(ode_prob)
n_t = length(sol.t) # number of discretization points in time span
grid = get_discrete(pde_sys, discretization)
grid_x = grid[x]
grid_v = grid[v(x, t)]
grid_w = grid[w(x, t)]
n_x = length(grid_x) # number of discretization points in x

using LinearAlgebra
snapshot_v = reduce(hcat, sol[grid_v])
snapshot_w = reduce(hcat, sol[grid_w])
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
display(svd_plt)

using CatViews # avoid copying and conveniently store unconnected data for plotting
split_vec_sizes = collect(isodd(i) ? (n_t,) : (1,) for i in 1:(2n_x))
vec_x, sol_x = splitview(split_vec_sizes...)
vec_v, sol_v = splitview(split_vec_sizes...)
vec_w, sol_w = splitview(split_vec_sizes...)
for (i, (xᵢ, vᵢ, wᵢ)) in enumerate(zip(grid_x, grid_v, grid_w))
    iₓ₂ = 2i
    iₓ₂₋₁ = iₓ₂ - 1
    sol_x[iₓ₂₋₁] .= xᵢ
    sol_x[iₓ₂] .= NaN # NaN is a path separator for plotting unconnected data
    sol_v[iₓ₂₋₁] .= sol[vᵢ]
    sol_v[iₓ₂] .= NaN
    sol_w[iₓ₂₋₁] .= sol[wᵢ]
    sol_w[iₓ₂] .= NaN
end
plt = plot(xlabel = L"v(x,t)", ylabel = L"x", zlabel = L"w(x,t)", xlims = (-0.5, 2.0),
           ylims = (0.0, L), zlims = (0.0, 0.25), legend = false, xflip = true,
           camera = (50, 30), titlefont = 10,
           title = "Phase−Space diagram of full $(nameof(pde_sys)) system")
for i in 1:2:(2n_x)
    plot!(plt, sol_v[i], sol_x[i], sol_w[i])
end
display(plt)

using ModelOrderReduction
snapshot_simpsys = Array(sol)
pod_dim = deim_dim = 5
pod_reducer = POD(snapshot_simpsys, pod_dim)
reduce!(pod_reducer, TSVD())
pod_basis = pod_reducer.rbasis
deim_sys = deim(simp_sys, pod_basis)
deim_prob = ODEProblem(deim_sys, nothing, tspan)
deim_sol = solve(deim_prob)
n_t_deim = length(deim_sol.t)

split_vec_sizes_deim = collect(isodd(i) ? (n_t_deim,) : (1,) for i in 1:(2n_x))
vec_deim_x, sol_deim_x = splitview(split_vec_sizes_deim...)
vec_deim_v, sol_deim_v = splitview(split_vec_sizes_deim...)
vec_deim_w, sol_deim_w = splitview(split_vec_sizes_deim...)
for (i, (xᵢ, vᵢ, wᵢ)) in enumerate(zip(grid_x, grid_v, grid_w))
    iₓ₂ = 2i
    iₓ₂₋₁ = iₓ₂ - 1
    sol_deim_x[iₓ₂₋₁] .= xᵢ
    sol_deim_x[iₓ₂] .= NaN
    sol_deim_v[iₓ₂₋₁] .= deim_sol[vᵢ]
    sol_deim_v[iₓ₂] .= NaN
    sol_deim_w[iₓ₂₋₁] .= deim_sol[wᵢ]
    sol_deim_w[iₓ₂] .= NaN
end

deim_plt = plot(xlabel = L"v(x,t)", ylabel = L"x", zlabel = L"w(x,t)", xlims = (-0.5, 2.0),
                ylims = (0.0, L), zlims = (0.0, 0.25), legend = false, xflip = true,
                camera = (50, 30), titlefont = 10,
                title = "Phase−Space diagram of reduced $(nameof(pde_sys)) system")
for i in 1:2:(2n_x)
    plot!(deim_plt, sol_deim_v[i], sol_deim_x[i], sol_deim_w[i])
end
display(deim_plt)

plt_2 = plot(xlabel = L"v(x,t)", ylabel = L"x", zlabel = L"w(x,t)", xlims = (-0.5, 2.0),
             ylims = (0.0, L), zlims = (0.0, 0.25), xflip = true,
             camera = (50, 30), titlefont = 10,
             title = "Comparison of full and reduced systems")
plot!(plt_2, vec_v, vec_x, vec_w, label = "Full$(length(ode_sys.eqs))")
plot!(plt_2, vec_deim_v, vec_deim_x, vec_deim_w, label = "POD$(pod_dim)/DEIM$(deim_dim)")
display(plt_2)
```