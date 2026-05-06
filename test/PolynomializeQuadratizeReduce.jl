using Test
using ModelOrderReduction
using ModelingToolkit
using OrdinaryDiffEq
using LinearAlgebra

using ModelingToolkit: t_nounits as t, D_nounits as D

@variables x(t) y(t)

eqs = [
    D(x) ~ -x + y + 0.1 * sqrt(x),
    D(y) ~ -2.0 * y + 0.2 * x^2,
]

@mtkcompile sys = System(eqs, t)

u0 = [1.0, 0.5]

u0_pairs = [
    x => 0.5,
    y => 1.0,
]

tspan = (0.0, 1.0)
saveat = 0.1
nmodes = 2

result = nothing

result = @test_nowarn(polynomialize_quadratize_reduce(
    sys,
    u0,
    tspan,
    nmodes;
    saveat = saveat,
    polynomialize_kwargs = (
        maxdepth = 6,
        maxnum = 10_000,
        new_var_base_name = "w_",
        start_new_vars_with = 0,
    ),
    quadratize_kwargs = (
        new_var_base_name = "z_",
        start_with = 0,
        max_depth = Inf,
    ),
))

@test result !== nothing

@test result.rom isa ModelingToolkit.System
@test result.polysys isa ModelingToolkit.System
@test result.quadsys isa ModelingToolkit.System

quad_unknowns = ModelingToolkit.unknowns(result.quadsys)
rom_unknowns = ModelingToolkit.unknowns(result.rom)

nquad = length(quad_unknowns)

@test length(result.a_vars) == nmodes
@test length(rom_unknowns) == nmodes
@test length(result.a0) == nmodes
@test length(result.a0_pairs) == nmodes

@test size(result.V) == (nquad, nmodes)
@test length(result.xbar) == nquad
@test result.V' * result.V ≈ Matrix{Float64}(I, nmodes, nmodes)

@test length(result.u0_poly) == length(ModelingToolkit.unknowns(result.polysys))
@test length(result.u0_poly_pairs) == length(ModelingToolkit.unknowns(result.polysys))

@test length(result.u0_quad) == nquad
@test length(result.u0_quad_pairs) == nquad
@test length(result.u0_quad_solver_order) == nquad

@test result.u0_quad_solver_order ≈ result.sol_quad.u[1]
@test result.a0 ≈ result.V' * (result.u0_quad_solver_order .- result.xbar)

@test result.poly_subs !== nothing
@test result.quad_subs !== nothing

@test string(result.sol_quad.retcode) == "Success"
@test string(result.sol_rom.retcode) == "Success"

@test length(result.sol_rom.u[1]) == nmodes
@test result.sol_rom.u[1] ≈ result.a0
@test result.sol_rom.t ≈ result.sol_quad.t

first_reconstruction = result.sol_rom[quad_unknowns[1]]

@test length(first_reconstruction) == length(result.sol_rom.t)
@test all(isfinite, first_reconstruction)