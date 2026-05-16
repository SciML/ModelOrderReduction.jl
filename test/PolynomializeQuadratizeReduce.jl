using Test
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
quadsys_raw, quad_subs = @test_nowarn polynomialize_and_quadratize(sys)

@test quadsys_raw !== nothing
@test quad_subs !== nothing
@test quadsys_raw isa ModelingToolkit.System

quadsys = ModelingToolkit.complete(quadsys_raw)

# Compute augmented initial conditions
u0_quad_pairs = compute_augmented_initial_pairs(
    sys,
    quadsys,
    u0_pairs,
    quad_subs,
)

@test length(u0_quad_pairs) == length(ModelingToolkit.unknowns(quadsys))

# Solve quadratized system
prob_quad = ODEProblem(quadsys, u0_quad_pairs, tspan)
sol_quad = solve(prob_quad, Tsit5(); saveat = saveat)

@test string(sol_quad.retcode) == "Success"

snapshots = reduce(hcat, sol_quad.u)

@test size(snapshots, 2) == length(sol_quad.t)

xbar = [sum(snapshots[i, :]) / size(snapshots, 2) for i in axes(snapshots, 1)]
centered_snapshots = snapshots .- reshape(xbar, :, 1)

# POD basis
F = svd(Matrix(centered_snapshots))
V = F.U[:, 1:nmodes]

nquad = length(ModelingToolkit.unknowns(quadsys))

@test size(V) == (nquad, nmodes)
@test V' * V ≈ Matrix{Float64}(I, nmodes, nmodes)

# Initial reduced coordinates
u0_quad_solver_order = sol_quad.u[1]
a0 = V' * (u0_quad_solver_order .- xbar)

iv = ModelingToolkit.get_iv(quadsys)

a_vars = [
    Symbolics.scalarize(@variables $(Symbol("a_", i))(iv))[1]
        for i in 1:nmodes
]

@test length(a_vars) == nmodes
@test length(a0) == nmodes

# Galerkin projection
rom_raw = galerkin_project_system_affine(
    quadsys,
    V,
    xbar,
    a_vars,
)

@test rom_raw isa ModelingToolkit.System

rom = ModelingToolkit.complete(rom_raw)

rom_unknowns = ModelingToolkit.unknowns(rom)

@test length(rom_unknowns) == nmodes

a0_pairs = [a_vars[i] => a0[i] for i in eachindex(a_vars)]

# Solve ROM
prob_rom = ODEProblem(rom, a0_pairs, tspan)
sol_rom = solve(prob_rom, Tsit5(); saveat = saveat)

@test string(sol_rom.retcode) == "Success"

@test sol_rom.t ≈ sol_quad.t
@test length(sol_rom.u[1]) == nmodes
@test sol_rom.u[1] ≈ a0

quad_unknowns = ModelingToolkit.unknowns(quadsys)
first_reconstruction = sol_rom[quad_unknowns[1]]

@test length(first_reconstruction) == length(sol_rom.t)
@test all(isfinite, first_reconstruction)
