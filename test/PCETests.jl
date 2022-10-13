using Test, ModelOrderReduction
using Symbolics, PolyChaos, ModelingToolkit, LinearAlgebra
const MOR = ModelOrderReduction

include("PCETestUtils.jl")

# testing extraction of independent variables 
@testset "PCE: get_independent_vars test" begin
    @variables t, z, u(t), v(t)[1:4], w(t, z), x(t, z)[1:4]
    @test isequal(MOR.get_independent_vars(u), [t])
    @test isequal(MOR.get_independent_vars(v[1]), [t])
    @test isequal(MOR.get_independent_vars(v[2]), [t])
    @test isequal(MOR.get_independent_vars(w), [t, z])
    @test isequal(MOR.get_independent_vars(x[2]), [t, z])
    @test isequal(MOR.get_independent_vars(collect(v)), [[t] for i in 1:length(v)])
    @test isequal(MOR.get_independent_vars(collect(x)), [[t, z] for i in 1:length(v)])
end

# test PCE generation
@testset "PCE: constructor test" begin
    @parameters a, b
    @variables y
    n = 5
    test_basis = [a => GaussOrthoPoly(n), b => Uniform01OrthoPoly(n + 1)]
    warn_message = "Currently only bases with identical degrees are supported." *
                   "\nProceed with minimum common degree = $n"
    @test_logs (:warn, warn_message) PCE([y], test_basis)

    test_basis = [a => GaussOrthoPoly(n), b => Uniform01OrthoPoly(n)]
    pce = PCE([y], test_basis)
    @test length(pce.moments[1]) == binomial(n + 2, 2)
    @test length(pce.sym_basis) == binomial(n + 2, 2)
    @test isequal(pce.parameters, [a, b])
end

# test equation for throughout:
@parameters a, b
@variables t, y(t)
D = Differential(t)
test_equation = [D(y) ~ a * y + 4 * b]

# set up pce
n = 5
bases = [a => GaussOrthoPoly(n)]
pce = PCE([y], bases)
eq = [eq.rhs for eq in test_equation]
pce_eq = MOR.apply_ansatz(eq, pce)[1]

@testset "PCE: apply_ansatz test" begin
    true_eq = expand(pce.sym_basis[2] * dot(pce.moments[1], pce.sym_basis) + 4 * b)
    @test isequal(pce_eq, true_eq)
end

# test extraction of monomial coefficients
coeffs = Dict{Any, Any}(pce.sym_basis[i] * pce.sym_basis[2] => pce.moments[1][i]
                        for i in 1:(n + 1))
coeffs[Val(1)] = 4.0 * b
basis_indices = Dict{Any, Any}(pce.sym_basis[i] * pce.sym_basis[2] => ([i - 1, 1],
                                                                       [1, i - 1])
                               for i in 1:(n + 1))
basis_indices[Val(1)] = [[0], [0]]

@testset "PCE: basismonomial extraction test" begin
    extracted_coeffs = MOR.extract_coeffs(pce_eq, pce.sym_basis)
    @test all(isequal(coeffs[mono], extracted_coeffs[mono]) for mono in keys(coeffs))

    extracted_coeffs, extracted_basis_indices = MOR.extract_basismonomial_coeffs([pce_eq],
                                                                                 pce)
    extracted_basis_indices = Dict(extracted_basis_indices)
    test1 = [isequal(basis_indices[mono][1], extracted_basis_indices[mono])
             for mono in keys(basis_indices)]
    test2 = [isequal(basis_indices[mono][2], extracted_basis_indices[mono])
             for mono in keys(basis_indices)]
    @test all(test1 + test2 .>= 1)
end

# test bump_degree
@testset "PCE: bump_degree test" begin
    n = 5
    n_bumped = 10
    shape_a = 0.1
    shape_b = 0.2
    shape = 0.3
    mu = 0.4
    λ = 0.5
    ϕ = 0.6
    rate = 1.0
    my_measure = Measure("my_measure", t -> 1 + t, (-1, 1), false, Dict())
    my_poly = OrthoPoly("my_poly", n, my_measure)
    orthogonal_polynomials = [
        my_poly => Dict(),
        GaussOrthoPoly(n) => [],
        Uniform01OrthoPoly(n) => [],
        Uniform_11OrthoPoly(n) => [],
        GammaOrthoPoly(n, shape, rate) => [shape, rate],
        HermiteOrthoPoly(n) => [],
        JacobiOrthoPoly(n, shape_a, shape_b) => [shape_a, shape_b],
        LaguerreOrthoPoly(n) => [],
        LogisticOrthoPoly(n) => [],
        MeixnerPollaczekOrthoPoly(n, λ, ϕ) => [λ, ϕ],
        genHermiteOrthoPoly(n, mu) => [mu],
        genLaguerreOrthoPoly(n, shape) => [shape],
        LegendreOrthoPoly(n) => [],
        Beta01OrthoPoly(n, shape_a, shape_b) => [shape_a, shape_b],
    ]

    for (op, params) in orthogonal_polynomials
        bumped_op = MOR.bump_degree(op, n_bumped)
        @test deg(bumped_op) == n_bumped
        @test MOR.measure_parameters(bumped_op.measure) == params
    end
end

# test Galerkin projection
@testset "PCE: galerkin projection test" begin
    moment_eqs = MOR.pce_galerkin(eq, pce)
    integrator = MOR.bump_degree(pce.pc_basis, n + 1)

    true_moment_eqs = Num[]
    for j in 0:n
        mom_eq = 0.0
        for mono in keys(basis_indices)
            ind = basis_indices[mono][2]
            c = computeSP(vcat(ind, j), integrator)
            mom_eq += c * coeffs[mono]
        end
        push!(true_moment_eqs, mom_eq)
    end

    @test integrator.deg == n + 1
    @test integrator.measure isa typeof(pce.pc_basis.measure)
    @test integrator.measure.measures[1] isa typeof(pce.pc_basis.measure.measures[1])
    @test all([isapprox_sym(moment_eqs[1][i], true_moment_eqs[i])
               for i in eachindex(true_moment_eqs)])

    # check generation of moment equations
    @named test_system = ODESystem(test_equation, t, [y], [a, b])
    moment_system, pce_eval = moment_equations(test_system, pce)
    moment_eqs = equations(moment_system)
    moment_eqs = [moment_eqs[i].rhs * computeSP([i - 1, i - 1], integrator)
                  for i in eachindex(moment_eqs)]
    @test isequal(parameters(moment_system), [b])
    @test nameof(moment_system) == :test_system_pce
    @test isequal(states(moment_system), reduce(vcat, pce.moments))
    @test all([isapprox_sym(moment_eqs[i], true_moment_eqs[i])
               for i in eachindex(true_moment_eqs)])
end
