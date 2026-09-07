using Test, ModelOrderReduction
import SymbolicIndexingInterface as SII
using ModelingToolkit, OrdinaryDiffEq
using LinearAlgebra: norm
using SciMLBase: successful_retcode
using ModelingToolkit: Symbolics, SymbolicUtils

function is_symbolic_array(expression)
    value = ModelingToolkit.Symbolics.unwrap(expression)
    return ModelingToolkit.SymbolicUtils.symtype(value) <: AbstractArray
end

function tree_size(expression)
    value = ModelingToolkit.Symbolics.unwrap(expression)
    ModelingToolkit.SymbolicUtils.iscall(value) || return 1
    return 1 + sum(
        tree_size, ModelingToolkit.SymbolicUtils.arguments(value); init = 0
    )
end

@independent_variables t
D = Differential(t)

@testset "array-equation POD Galerkin" begin
    @variables z(t)[1:4]
    z_scalars = Symbolics.value.(Symbolics.scalarize(z))
    z0 = [0.4, 0.3, 0.2, 0.1]
    @named array_system = System(
        [D(z) ~ -z - z .^ 3], t, z_scalars, [];
        initial_conditions = [z => z0, D(z) => -z0 - z0 .^ 3],
    )
    full_problem = DAEProblem(
        complete(array_system), nothing, (0.0, 0.5); build_initializeprob = false
    )
    full_solution = solve(full_problem; saveat = 0.05, abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(full_solution)

    rom = pod(full_problem, full_solution, 2)
    @test length(ModelingToolkit.get_eqs(rom)) == 1
    @test is_symbolic_array(only(ModelingToolkit.get_eqs(rom)).lhs)
    @test is_symbolic_array(only(ModelingToolkit.get_eqs(rom)).rhs)
    @test length(ModelingToolkit.get_observed(rom)) == 1
    @test is_symbolic_array(only(ModelingToolkit.get_observed(rom)).lhs)

    rom_problem = DAEProblem(
        rom, nothing, full_problem.tspan; build_initializeprob = false
    )
    rom_solution = solve(rom_problem; saveat = 0.05, abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(rom_solution)
    truth = Array(full_solution)
    reconstruction = reduce(
        hcat, [
            SII.observed(rom, z)(state, rom_problem.p, time)
                for (state, time) in zip(rom_solution.u, rom_solution.t)
        ]
    )
    @test norm(reconstruction - truth) / norm(truth) < 0.01
end

@testset "POD graph size is grid-independent for array equations" begin
    function graph_size(n)
        @variables q(t)[1:n]
        q_scalars = Symbolics.value.(Symbolics.scalarize(q))
        @named sys = System([D(q) ~ -q - q .^ 3], t, q_scalars, [])
        local_snapshot = [sin(0.3 * i * j) + cos(0.2 * i * (j + 1)) for i in 1:n, j in 1:8]
        rom = pod(complete(sys), local_snapshot, 2)
        equation = only(ModelingToolkit.get_eqs(rom))
        return tree_size(equation.lhs) + tree_size(equation.rhs)
    end
    @test graph_size(4) == graph_size(12)
end

@testset "scalarized POD Galerkin fallback" begin
    @variables r(t)[1:4]
    r0 = [0.4, 0.3, 0.2, 0.1]
    @mtkcompile ode_source = System([D(r) ~ -r - r .^ 3], t)
    ode_problem = ODEProblem(ode_source, [r => r0], (0.0, 0.5))
    ode_solution = solve(
        ode_problem, Tsit5(); saveat = 0.05, abstol = 1.0e-10, reltol = 1.0e-10
    )
    @test successful_retcode(ode_solution)
    rom = pod(ode_problem, ode_solution, 2)
    @test length(ModelingToolkit.get_eqs(rom)) == 1
    @test is_symbolic_array(only(ModelingToolkit.get_eqs(rom)).rhs)

    truth = reduce(hcat, ode_solution[r])
    dae_problem = DAEProblem(rom, nothing, ode_problem.tspan; build_initializeprob = false)
    dae_solution = solve(dae_problem; saveat = 0.05, abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(dae_solution)
    reconstruction = reduce(
        hcat, [
            SII.observed(rom, r)(state, dae_problem.p, time)
                for (state, time) in zip(dae_solution.u, dae_solution.t)
        ]
    )
    @test norm(reconstruction - truth) / norm(truth) < 0.01
end

@testset "parameterized array POD" begin
    @parameters a = 2.0
    @variables z(t)[1:4] energy(t)
    z_scalars = Symbolics.value.(Symbolics.scalarize(z))
    z0 = [0.4, -0.2, 0.3, 0.1]
    @named parameterized = System(
        [D(z) ~ -z - a * (z .^ 3)], t, z_scalars, [a];
        observed = [energy ~ sum(z)],
        initial_conditions = [z => z0, D(z) => -z0 - 3 .* z0 .^ 3],
    )
    full_problem = DAEProblem(
        complete(parameterized), [a => 3.0], (0.0, 0.5); build_initializeprob = false
    )
    full_solution = solve(full_problem; saveat = 0.05, abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(full_solution)
    rom = pod(full_problem, full_solution, 2)
    rom_problem = DAEProblem(
        rom, nothing, full_problem.tspan; build_initializeprob = false
    )
    @test rom_problem.ps[a] == 3.0
    rom_solution = solve(rom_problem; saveat = 0.05, abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(rom_solution)
    @test length(rom_solution[energy]) == length(full_solution.t)
    truth = Array(full_solution)
    reconstruction = reduce(
        hcat, [
            SII.observed(rom, z)(state, rom_problem.p, time)
                for (state, time) in zip(rom_solution.u, rom_solution.t)
        ]
    )
    @test norm(reconstruction - truth) / norm(truth) < 0.01
end
