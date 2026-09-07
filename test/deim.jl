using Test, ModelOrderReduction
import SymbolicIndexingInterface as SII
using ModelingToolkit, MethodOfLines, OrdinaryDiffEq
using LinearAlgebra: norm, svd
using SciMLBase: successful_retcode, symbolic_discretize

# construct an ModelingToolkit.System with non-empty field substitutions
@independent_variables x t
@variables v(..) w(..)
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
eqs = [
    ε * Dt(v(x, t)) ~ ε^2 * Dxx(v(x, t)) + f(v(x, t)) - w(x, t) + c,
    Dt(w(x, t)) ~ b * v(x, t) - γ * w(x, t) + c,
]
bcs = [
    v(x, 0.0) ~ 0.0,
    w(x, 0) ~ 0.0,
    Dx(v(0, t)) ~ -i₀(t),
    Dx(v(L, t)) ~ 0.0,
]
domains = [
    x ∈ (0.0, L),
    t ∈ (0.0, 14.0),
]
ivs = [x, t]
dvs = [v(x, t), w(x, t)]
pde_sys = PDESystem(eqs, bcs, domains, ivs, dvs; name = :FN)

N = 5 # (minimum number of) equidistant discretization intervals
dx = (L - 0.0) / N
dxs = [x => dx]
order = 2
discretization = MOLFiniteDifference(dxs, t; approx_order = order)
full_prob = discretize(pde_sys, discretization; fallback = false)
@test full_prob isa DAEProblem
sol = solve(full_prob; saveat = 1.0)
@test successful_retcode(sol)

snapshot = Array(sol.original_sol)
pod_dim = 3
deim_sys = @test_nowarn deim(full_prob, sol, pod_dim)

# check the number of dependent variables in the new system
@test length(ModelingToolkit.get_unknowns(deim_sys)) == pod_dim

function is_symbolic_array(expression)
    value = ModelingToolkit.Symbolics.unwrap(expression)
    return ModelingToolkit.SymbolicUtils.symtype(value) <: AbstractArray
end

reduced_equations = ModelingToolkit.get_eqs(deim_sys)
@test length(reduced_equations) == 1
@test is_symbolic_array(only(reduced_equations).lhs)
reconstruction_equations = ModelingToolkit.get_observed(deim_sys)
@test length(reconstruction_equations) == length(dvs)
@test all(reconstruction_equations) do equation
    is_symbolic_array(equation.lhs) && is_symbolic_array(equation.rhs)
end
reconstruction_parameters = filter(ModelingToolkit.get_ps(deim_sys)) do parameter
    is_symbolic_array(parameter) &&
        size(ModelingToolkit.Symbolics.wrap(parameter)) == (size(snapshot, 1), pod_dim)
end
@test length(reconstruction_parameters) == 1

deim_prob = DAEProblem(
    deim_sys, nothing, full_prob.tspan;
    build_initializeprob = false,
)

deim_sol = solve(deim_prob; saveat = 1.0)
@test successful_retcode(deim_sol)

nₓ = length(sol[x])
nₜ = length(sol[t])

# test solution retrieval
@test size(deim_sol[v(x, t)]) == (nₓ, nₜ)
@test size(deim_sol[w(x, t)]) == (nₓ, nₜ)
full_fields = vcat(sol[v(x, t)], sol[w(x, t)])
reduced_fields = vcat(deim_sol[v(x, t)], deim_sol[w(x, t)])
@test norm(reduced_fields - full_fields) / norm(full_fields) < 0.6

@parameters a = 2.0
@variables z(t)[1:4] energy(t)
z_scalars = ModelingToolkit.Symbolics.value.(
    ModelingToolkit.Symbolics.scalarize(z)
)
@named parameterized_array_system = System(
    [Dt(z) ~ -z - a * (z .^ 3)], t, z_scalars, [a];
    observed = [energy ~ sum(z)],
)
parameterized_snapshot = [
    sin(0.3 * i * j) + cos(0.2 * i * (j + 1)) for i in 1:4, j in 1:8
]
@test_throws DimensionMismatch deim(
    parameterized_array_system, parameterized_snapshot, 2; snapshot_times = [0.0]
)
parameterized_reduced = deim(parameterized_array_system, parameterized_snapshot, 2)
parameterized_prob = DAEProblem(
    parameterized_reduced, nothing, (0.0, 1.0);
    build_initializeprob = false,
)
@test parameterized_prob.ps[a] == 2.0
lift_parameter = only(
    filter(ModelingToolkit.parameters(parameterized_reduced)) do p
        is_symbolic_array(p) && size(ModelingToolkit.Symbolics.wrap(p)) == (4, 2)
    end
)
lift = parameterized_prob.ps[lift_parameter]
nonlinear_basis = svd(-2.0 .* parameterized_snapshot .^ 3).U[:, 1:2]
first_index = argmax(abs.(nonlinear_basis[:, 1]))
second_residual = nonlinear_basis[:, 2] - nonlinear_basis[:, 1] *
    (nonlinear_basis[first_index, 2] / nonlinear_basis[first_index, 1])
indices = [first_index, argmax(abs.(second_residual))]
for state in ([0.1, -0.2], [-0.3, 0.4])
    full_state = lift * state
    expected = -state + lift' * nonlinear_basis *
        (nonlinear_basis[indices, :] \ (-2.0 .* full_state[indices] .^ 3))
    residual = zeros(2)
    parameterized_prob.f(residual, expected, state, parameterized_prob.p, 0.0)
    @test norm(residual) < 1.0e-10
end

parameterized_sol = solve(parameterized_prob; saveat = 0.2)
@test successful_retcode(parameterized_sol)
@test length(parameterized_sol[energy]) == 6
completed_array_reduced = deim(
    complete(deepcopy(parameterized_array_system)), parameterized_snapshot, 2
)
@test length(ModelingToolkit.get_eqs(completed_array_reduced)) == 1
@test is_symbolic_array(only(ModelingToolkit.get_eqs(completed_array_reduced)).lhs)
@test is_symbolic_array(only(ModelingToolkit.get_eqs(completed_array_reduced)).rhs)

# `mtkcompile` scalarizes this array equation and reverses its registered element order.
# DEIM must still emit one correctly ordered array equation.
@variables compiled_z(t)[1:4]
@mtkcompile scalarized_array_system = System(
    [Dt(compiled_z) ~ -compiled_z - compiled_z .^ 3], t;
    name = :scalarized_array_system,
)
@test length(ModelingToolkit.get_eqs(scalarized_array_system)) == 4
@test all(ModelingToolkit.get_eqs(scalarized_array_system)) do equation
    !is_symbolic_array(equation.lhs) && !is_symbolic_array(equation.rhs)
end
scalarized_snapshot = [
    (0.2 + 0.1 * i) * exp(-0.1 * j) + (1.0 - 0.05 * i) * sin(0.2 * j)
        for i in 1:4, j in 1:8
]
array_codegen_reduced = deim(scalarized_array_system, scalarized_snapshot, 2)
array_codegen_equations = ModelingToolkit.get_eqs(array_codegen_reduced)
@test length(array_codegen_equations) == 1
@test is_symbolic_array(only(array_codegen_equations).lhs)
@test is_symbolic_array(only(array_codegen_equations).rhs)
array_codegen_observed = ModelingToolkit.get_observed(array_codegen_reduced)
@test length(array_codegen_observed) == 1
@test is_symbolic_array(only(array_codegen_observed).lhs)
@test is_symbolic_array(only(array_codegen_observed).rhs)
array_codegen_prob = DAEProblem(
    array_codegen_reduced, nothing, (0.0, 1.0);
    build_initializeprob = false,
)
array_codegen_sol = solve(array_codegen_prob; saveat = 0.2)
@test successful_retcode(array_codegen_sol)
@test size(Array(array_codegen_sol)) == (2, 6)
array_observed_function = SII.observed(array_codegen_reduced, compiled_z)
initial_reconstruction = array_observed_function(
    array_codegen_prob.u0, array_codegen_prob.p, first(array_codegen_prob.tspan)
)
@test initial_reconstruction ≈
    reverse(scalarized_snapshot[:, 1])

# Multidimensional arrays may be scalarized into a noncontiguous permutation rather than a
# simple reversal. Reconstruction must restore their canonical Cartesian index order.
@variables compiled_matrix(t)[1:2, 1:3]
@mtkcompile scalarized_matrix_system = System(
    [Dt(compiled_matrix) ~ -compiled_matrix - compiled_matrix .^ 3], t;
    name = :scalarized_matrix_system,
)
matrix_unknowns = ModelingToolkit.get_unknowns(scalarized_matrix_system)
matrix_rows_by_index = Dict{Tuple, Int}()
for (row, unknown) in enumerate(matrix_unknowns)
    index_arguments = ModelingToolkit.SymbolicUtils.arguments(
        ModelingToolkit.Symbolics.unwrap(unknown)
    )
    index = Tuple(
        ModelingToolkit.SymbolicUtils.unwrap_const.(index_arguments[2:end])
    )
    matrix_rows_by_index[index] = row
end
matrix_canonical_rows = vec(
    [
        matrix_rows_by_index[Tuple(index)] for index in CartesianIndices((2, 3))
    ]
)
@test length(unique(diff(matrix_canonical_rows))) > 1
matrix_snapshot = [
    (0.3 + 0.07 * i) * exp(-0.1 * j) + (0.8 - 0.03 * i) * sin(0.2 * j)
        for i in 1:6, j in 1:8
]
matrix_reduced = deim(scalarized_matrix_system, matrix_snapshot, 2)
matrix_equations = ModelingToolkit.get_eqs(matrix_reduced)
@test length(matrix_equations) == 1
@test is_symbolic_array(only(matrix_equations).lhs)
@test is_symbolic_array(only(matrix_equations).rhs)
matrix_prob = DAEProblem(
    matrix_reduced, nothing, (0.0, 1.0);
    build_initializeprob = false,
)
matrix_sol = solve(matrix_prob; saveat = 0.2)
@test successful_retcode(matrix_sol)
matrix_observed_function = SII.observed(matrix_reduced, compiled_matrix)
initial_matrix_reconstruction = matrix_observed_function(
    matrix_prob.u0, matrix_prob.p, first(matrix_prob.tspan)
)
@test initial_matrix_reconstruction ≈
    reshape(matrix_snapshot[matrix_canonical_rows, 1], 2, 3)

function tree_size(expression)
    value = ModelingToolkit.Symbolics.unwrap(expression)
    ModelingToolkit.SymbolicUtils.iscall(value) || return 1
    return 1 + sum(
        tree_size, ModelingToolkit.SymbolicUtils.arguments(value); init = 0
    )
end

function reduced_system_size(intervals)
    local_dx = L / intervals
    local_discretization = MOLFiniteDifference(
        [x => local_dx], t; approx_order = order
    )
    local_sys, _ = symbolic_discretize(pde_sys, local_discretization)
    rows = length(ModelingToolkit.get_unknowns(local_sys))
    local_snapshot = [
        sin(0.13 * i * j) + cos(0.07 * i * (j + 1)) for i in 1:rows, j in 1:8
    ]
    local_reduced = deim(local_sys, local_snapshot, 2; snapshot_times = collect(0.0:0.1:0.7))
    local_equations = ModelingToolkit.get_eqs(local_reduced)
    local_observed = ModelingToolkit.get_observed(local_reduced)
    return (
        equations = length(local_equations),
        equation_tree = sum(local_equations; init = 0) do equation
            tree_size(equation.lhs) + tree_size(equation.rhs)
        end,
        observed = length(local_observed),
        observed_tree = sum(local_observed; init = 0) do equation
            tree_size(equation.lhs) + tree_size(equation.rhs)
        end,
        array_parameters = count(
            is_symbolic_array, ModelingToolkit.get_ps(local_reduced)
        ),
    )
end

# Fixed POD/DEIM dimensions produce a grid-independent generated symbolic graph.
@test reduced_system_size(5) == reduced_system_size(20)

@testset "training parameters and nonautonomous nonlinearities" begin
    @parameters α = 2.0
    @variables q(t)[1:4]
    q0 = [0.4, 0.3, 0.2, 0.1]
    qstates = ModelingToolkit.Symbolics.unwrap.(ModelingToolkit.Symbolics.scalarize(q))
    @named training_system = System(
        [Dt(q) ~ -q - α * (1 + sin(t)) * q .^ 3], t, qstates, [α];
        initial_conditions = [
            q => q0, Dt(q) => -q0 - 3 .* q0 .^ 3,
        ]
    )
    training_problem = DAEProblem(
        complete(training_system), [α => 3.0], (0.0, 0.5); build_initializeprob = false
    )
    training_solution = solve(training_problem; saveat = 0.05, abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(training_solution)
    trained = deim(training_problem, training_solution, 2)
    trained_problem = DAEProblem(trained, nothing, training_problem.tspan; build_initializeprob = false)
    @test trained_problem.ps[α] == 3.0
    trained_solution = solve(trained_problem; saveat = 0.05, abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(trained_solution)
    truth = Array(training_solution)
    reconstruct = SII.observed(trained, q)
    reconstruction = reduce(
        hcat, [
            reconstruct(state, trained_problem.p, time)
                for (state, time) in zip(trained_solution.u, trained_solution.t)
        ]
    )
    @test norm(reconstruction - truth) / norm(truth) < 0.01
end

@testset "ODEProblem sources and scalar code generation" begin
    @variables r(t)[1:4]
    r0 = [0.4, 0.3, 0.2, 0.1]
    @mtkcompile ode_source = System([Dt(r) ~ -r - r .^ 3], t)
    ode_problem = ODEProblem(ode_source, [r => r0], (0.0, 0.5))
    ode_solution = solve(ode_problem, Tsit5(); saveat = 0.05, abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(ode_solution)
    rom = deim(ode_problem, ode_solution, 2)
    @test length(ModelingToolkit.get_eqs(rom)) == 1
    @test is_symbolic_array(only(ModelingToolkit.get_eqs(rom)).rhs)
    # `mtkcompile` permutes the scalarized unknowns; index by `r` for canonical order.
    truth = reduce(hcat, ode_solution[r])

    dae_problem = DAEProblem(rom, nothing, ode_problem.tspan; build_initializeprob = false)
    dae_solution = solve(dae_problem; saveat = 0.05, abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(dae_solution)
    dae_reconstruction = reduce(
        hcat, [
            SII.observed(rom, r)(state, dae_problem.p, time)
                for (state, time) in zip(dae_solution.u, dae_solution.t)
        ]
    )
    @test norm(dae_reconstruction - truth) / norm(truth) < 0.01

    compiled = mtkcompile(rom)
    @test length(ModelingToolkit.get_eqs(compiled)) == 2
    @test length(ModelingToolkit.get_observed(compiled)) == 1
    scalar_problem = ODEProblem(compiled, nothing, ode_problem.tspan; build_initializeprob = false)
    scalar_solution = solve(scalar_problem, Tsit5(); saveat = 0.05, abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(scalar_solution)
    scalar_reconstruction = reduce(
        hcat, [
            SII.observed(compiled, r)(state, scalar_problem.p, time)
                for (state, time) in zip(scalar_solution.u, scalar_solution.t)
        ]
    )
    @test norm(scalar_reconstruction - dae_reconstruction) / norm(truth) < 1.0e-6
end

@testset "nonautonomous linear terms" begin
    # Two distinct decay rates give rank-two state and nonlinear snapshots, so two POD
    # and DEIM modes reproduce the full model up to solver tolerance.
    @variables s(t)[1:6]
    s0 = [0.4, 0.3, 0.2, 0.1, -0.2, 0.5]
    rates = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    s_scalars = ModelingToolkit.Symbolics.unwrap.(ModelingToolkit.Symbolics.scalarize(s))
    @named forced_system = System(
        [Dt(s) ~ -(rates .+ sin(t)) .* s], t, s_scalars, [];
        initial_conditions = [s => s0, Dt(s) => -rates .* s0],
    )
    forced_problem = DAEProblem(
        complete(forced_system), nothing, (0.0, 0.5); build_initializeprob = false
    )
    forced_solution = solve(forced_problem; saveat = 0.05, abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(forced_solution)
    fom = ModelOrderReduction.FullOrderModel(forced_problem.f.sys)
    @test all(iszero, fom.forcing)
    @test all(!iszero, fom.nonlinear)
    rom = deim(forced_problem, forced_solution, 2)
    rom_problem = DAEProblem(rom, nothing, forced_problem.tspan; build_initializeprob = false)
    rom_solution = solve(rom_problem; saveat = 0.05, abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(rom_solution)
    truth = Array(forced_solution)
    reconstruction = reduce(
        hcat, [
            SII.observed(rom, s)(state, rom_problem.p, time)
                for (state, time) in zip(rom_solution.u, rom_solution.t)
        ]
    )
    @test norm(reconstruction - truth) / norm(truth) < 1.0e-6
end

@testset "the reduced system carries a tspan" begin
    @variables p(t)[1:6]
    p0 = [0.4, 0.3, 0.2, 0.1, -0.2, 0.5]
    pstates = ModelingToolkit.Symbolics.unwrap.(ModelingToolkit.Symbolics.scalarize(p))
    span = (0.0, 0.5)
    @named source = System(
        [Dt(p) ~ -p - p .^ 3], t, pstates, [];
        initial_conditions = [p => p0, Dt(p) => -p0 - p0 .^ 3],
    )
    source_problem = DAEProblem(complete(source), nothing, span; build_initializeprob = false)
    source_solution = solve(source_problem; saveat = 0.05, abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(source_solution)

    # The problem method trains on `prob`, so its interval is the reduced model's.
    from_problem = deim(source_problem, source_solution, 3)
    @test ModelingToolkit.get_tspan(from_problem) == span
    reduced_problem = DAEProblem(from_problem, nothing; build_initializeprob = false)
    @test reduced_problem.tspan == span
    reduced_solution = solve(reduced_problem; saveat = 0.05, abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(reduced_solution)
    truth = reduce(hcat, source_solution[p])
    reconstruction = reduce(
        hcat, [
            SII.observed(from_problem, p)(state, reduced_problem.p, time)
                for (state, time) in zip(reduced_solution.u, reduced_solution.t)
        ]
    )
    @test norm(reconstruction - truth) / norm(truth) < 1.0e-4

    # The system method inherits whatever the source system stores.
    snapshot = [sin(0.3 * i * j) + cos(0.2 * i * (j + 1)) for i in 1:6, j in 1:8]
    @mtkcompile with_span = System([Dt(p) ~ -p - p .^ 3], t; tspan = span)
    @test ModelingToolkit.get_tspan(deim(with_span, snapshot, 2)) == span

    # A source system without one must not acquire a fabricated interval.
    @mtkcompile without_span = System([Dt(p) ~ -p - p .^ 3], t)
    bare = deim(without_span, snapshot, 2)
    @test ModelingToolkit.get_tspan(bare) === nothing
    @test DAEProblem(bare, nothing, span; build_initializeprob = false).tspan == span
end
