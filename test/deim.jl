using Test
using ModelOrderReduction
using SparseArrays: sprand
using ModelingToolkit
using LinearAlgebra: svd

const N = 10
@variables t u[1:N](t)
A = sprand(N, N, 0.3)
D = Differential(t)
f(x) = sin(x)
@named sys = ODESystem(D.(u) .~ A * u + f.(u))
const n_snapshot = 2N
const pod_dim = 5
pod_basis = @view svd(rand(N, n_snapshot)).U[:, 1:pod_dim]
deim_sys = deim(sys, pod_basis)
@test length(states(deim_sys)) == pod_dim
@test length(equations(deim_sys)) == pod_dim
