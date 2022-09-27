using ModelOrderReduction, Plots, ModelingToolkit, DifferentialEquations, PolyChaos

# Reaction system
# k[1](1+θ[1]/2)/k[2]: A + B ⇌ C 
# k[3](1+θ[2]/2):      C → D
# k[4]:                B → D
# 
# state: c[1:4] = [A, B, C, D]
# certain parameters: k[1:4]
# uncertain parameters: θ[1:2] ∼ U[-1,1] 

@parameters k[1:4], θ[1:2]
@variables t, c(t)[1:4]
D = Differential(t)
reactor = [D(c[1]) ~ -k[1]*(1+0.5*θ[1])*c[1]*c[2] + k[2]*c[3];
           D(c[2]) ~ -k[1]*(1+0.5*θ[1])*c[1]*c[2] - k[4]*c[2] + k[2]*c[3];
           D(c[3]) ~ k[1]*c[1]*c[2] - k[3]*c[3];
           D(c[4]) ~ k[3]*(1+0.5*θ[2])*c[3] + k[4]*c[2]];

@named reactor_model = ODESystem(reactor, t, c, vcat(k, θ))

# 1. choose/generate PCE of the system state
d_pce = 3
pce = PCE(c, [θ[i] => Uniform_11OrthoPoly(d_pce) for i in eachindex(θ)])

# 2. generate moment equations
moment_eqs, pce_eval = moment_equations(reactor_model, pce)

# 3. solve the moment equations
# 3a. find initial moments via Galerkin projection (note that c0 could depend on an uncertain parameter)
c0 = [1.0, 2.0, 0.0, 0.0]
z0 = reduce(vcat, pce_galerkin(c0, pce))

# 3b. solve the initial value problem for the moment equations
moment_prob = ODEProblem(moment_eqs, z0, (0.0, 10.0), [k[1] => 1.0, k[2] => 0.2, k[3] => 12, k[4] => 0.1])
moment_sol = solve(moment_prob, Tsit5())

# 3c. define a function approximating the parametric solution to the IVP.
pce_sol(t, θ) = pce_eval(moment_sol(t), θ)

# 4. now let's compare to true solution
# take 100 uniform samples
n = 5
θsamples = Iterators.product(range(-1, 1, length=n), range(-1,1,length=n))
ps = [k[1] => 1.0, k[2] => 0.2, k[3] => 12, k[4] => 0.1, θ[1] => 0.0, θ[2] => 0.0]
pvals = [p[2] for p in ps]
reactor_problem = ODEProblem(reactor_model, c0, (0.0, 10.0), ps)
t_range = range(0.0, 10.0, length = 100)
sols = []
for θval in θsamples
    pvals[end-1] = θval[1]
    pvals[end] = θval[2]
    _prob = remake(reactor_problem, p = pvals)
    push!(sols, Array(solve(_prob, Tsit5(), saveat=t_range)))
end

species = ["A", "B", "C", "D"]
color = [:red, :green, :blue, :orange]
plots = [plot(title="Species $(species[i])", xlabel = "time", ylabel = "concentration", legend = false) for i in 1:4]
for sol in sols
    for i in 1:4
        plot!(plots[i], t_range, sol[i,:], color = color[i])
    end
end
for θval in θsamples
    pce_predictions = [pce_sol(t, [θval...]) for t in t_range]
    for i in 1:4
        plot!(plots[i], t_range, [c[i] for c in pce_predictions], color = "black", linestyle=:dash)
    end
end

fig = plot(plots...)
savefig(fig,string(@__DIR__,"/traces.png"))

n = 100
θsamples = Iterators.product(range(-1, 1, length=n), range(-1,1,length=n))
ps = [k[1] => 1.0, k[2] => 0.2, k[3] => 12, k[4] => 0.1, θ[1] => 0.0, θ[2] => 0.0]
pvals = [p[2] for p in ps]
reactor_problem = ODEProblem(reactor_model, c0, (0.0, 10.0), ps)
t_range = range(0.0, 10.0, length = 100)
sols = []
for θval in θsamples
    pvals[end-1] = θval[1]
    pvals[end] = θval[2]
    _prob = remake(reactor_problem, p = pvals)
    push!(sols, Array(solve(_prob, Tsit5(), saveat=t_range)))
end

mean_solution = mean(sols)
var_solution = mean([sol.^2 for sol in sols]) - mean_solution.^2

L = length(pce.sym_basis)
mean_PCE = [[moment_sol(t)[(i-1)*L+1:i*L][1] for i in 1:4] for t in t_range]
var_weightings = computeSP2(pce.pc_basis)
var_PCE =  [[dot(moment_sol(t)[(i-1)*L+1:i*L] .^ 2, var_weightings) for i in 1:4] for t in t_range] .- [m .^2 for m in mean_PCE]

species = ["A", "B", "C", "D"]
color = [:red, :green, :blue, :orange]
plots = [plot(title="Species $(species[i])", xlabel = "time", ylabel = "mean", legend = false) for i in 1:4]

for i in 1:4
    plot!(plots[i], t_range, [mean_solution[i,k] for k in axes(mean_solution,2)], color = color[i])
    plot!(plots[i], t_range, [s[i] for s in mean_PCE], color = "black", linestyle=:dash)
end
fig = plot(plots...)
savefig(fig, string(@__DIR__,"/mean.png"))

plots = [plot(title="Species $(species[i])", xlabel = "time", ylabel = "variance", legend = false) for i in 1:4]
for i in 1:4
    plot!(plots[i], t_range, [var_solution[i,k] for k in axes(var_solution,2)], color = color[i])
    plot!(plots[i], t_range, [s[i] for s in var_PCE], color = "black", linestyle=:dash)
end
fig = plot(plots...)
savefig(fig, string(@__DIR__,"/var.png"))
