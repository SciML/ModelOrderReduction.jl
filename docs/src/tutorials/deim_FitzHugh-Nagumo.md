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
    t ∈ (0.0, 7.0)]
ivs = [x, t]
dvs = [v(x, t), w(x, t)]
pdesys = PDESystem(eqs, bcs, domains, ivs, dvs; name = Symbol("FitzHugh-Nagumo"))

using MethodOfLines
N = 15 # equidistant discretization intervals
dx = (L - 0.0) / N
dxs = [x => dx]
order = 2
discretization = MOLFiniteDifference(dxs, t; approx_order = order)
odesys, tspan = symbolic_discretize(pdesys, discretization)
simpsys = structural_simplify(odesys)
odeprob = ODEProblem(simpsys, nothing, tspan)

using DifferentialEquations
sol = solve(odeprob)
grid = get_discrete(pdesys, discretization)
grid_x = grid[x]
grid_v = grid[v(x, t)]
grid_w = grid[w(x, t)]

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

plt = plot(xlabel = L"v(x,t)", ylabel = L"x", zlabel = L"w(x,t)", xlims = (-0.5, 2.0),
           ylims = (0.0, L), zlims = (0.0, 0.25), legend = false, xflip = true,
           camera = (50, 30), titlefont = 10,
           title = "Phase−Space diagram of full $(nameof(pdesys)) system")
for (xᵢ, vᵢ, wᵢ) in zip(grid_x, grid_v, grid_w)
    plot!(plt, sol[vᵢ], _ -> xᵢ, sol[wᵢ])
end
display(plt)

snapshot_simpsys = Array(sol)
```