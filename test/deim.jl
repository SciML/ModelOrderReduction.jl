using Test
using ModelOrderReduction
using SparseArrays
using ModelingToolkit
using LinearAlgebra

const N = 10 # number of dependent variables
@variables t u[1:N](t)
A = sprand(N, N, 0.3)
D = Differential(t)
f(x) = sin(x)
@named sys = ODESystem(D.(u) .~ A * u + f.(u))
const n_snapshot = 2N
const pod_dim = 5
pod_basis = @view svd(rand(N, n_snapshot)).U[:, 1:pod_dim]
sys_deim = deim(sys, pod_basis)
@test length(states(sys_deim)) == pod_dim
@test length(equations(sys_deim)) == pod_dim
@test length(observed(sys_deim)) == length(observed(sys)) + N

u0 = Dict(collect(u) .=> rand(N))
tspan = (0.0, 1.0)
prob = ODEProblem(sys, u0, tspan)
prob_deim = ODEProblem(sys_deim, u0, tspan)
@test pod_basis' * prob.u0 ≈ prob_deim.u0 # check projection for initial values
