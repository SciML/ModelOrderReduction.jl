using Test, ModelOrderReduction
using ModelingToolkit
using ModelingToolkit: Symbolics

@independent_variables t
@variables w(t) x(t) y(t) z(t)
@parameters α
D = Differential(t)

symbolically_zero(expression) = iszero(simplify(expand(expression)))

@testset "separate_terms" begin
    @testset "term classification" begin
        vars = [x, y, z]
        exprs = [2x + 3.0, sin(t) + x * sin(t), α * y + x * (1 - y), 4w]
        A, forcing, nonlinear = ModelOrderReduction.separate_terms(exprs, vars)
        @test A isa AbstractMatrix{Float64}
        @test Matrix(A) == [2 0 0; 0 0 0; 1 0 0; 0 0 0]
        @test symbolically_zero(forcing[1] - 3.0)
        @test iszero(nonlinear[1])
        @test symbolically_zero(forcing[2] - sin(t))
        @test symbolically_zero(nonlinear[2] - x * sin(t))
        @test iszero(forcing[3])
        @test symbolically_zero(nonlinear[3] - (α * y - x * y))
        @test symbolically_zero(forcing[4] - 4w)
        @test iszero(nonlinear[4])
    end

    @testset "zero expressions" begin
        vars = [x, y, z]
        exprs = fill(Num(0), 4)
        A, forcing, nonlinear = ModelOrderReduction.separate_terms(exprs, vars)
        @test size(A) == (4, 3)
        @test iszero(A)
        @test all(iszero, forcing)
        @test all(iszero, nonlinear)
    end

    @testset "nonunique vars" begin
        @test_throws ArgumentError ModelOrderReduction.separate_terms([x + y], [x, y, y])
    end
end

@testset "FullOrderModel" begin
    @variables u(t)[1:2, 1:2] v(t)
    @parameters β = 3.0
    u_scalars = Symbolics.unwrap.(vec(Symbolics.scalarize(u)))
    @named sys = System(
        [D(u) ~ -β * u .^ 2 .+ sin(t), D(v) ~ v + u[1, 1]], t,
        [u_scalars; Symbolics.unwrap(v)], [β]; observed = [w ~ v + sum(u)]
    )
    fom = ModelOrderReduction.FullOrderModel(complete(sys))
    @test length(fom.source_unknowns) == 5
    @test length(fom.unknowns) == 5
    @test sort(fom.rows) == 1:5
    @test all(isequal.([field.variable for field in fom.fields], Symbolics.unwrap.([u, v])))
    @test fom.fields[1].shape == (2, 2)
    @test fom.fields[2].shape == ()
    @test all(isequal.(fom.source_unknowns[fom.fields[1].rows], u_scalars))
    @test isequal(collect(keys(fom.parameter_values)), [Symbolics.unwrap(β)])
    @test collect(values(fom.parameter_values)) == [3.0]
    @test length(fom.observed) == 1
    @test isequal(only(fom.observed).lhs, w)

    row_of(variable) = findfirst(isequal(Symbolics.unwrap(variable)), fom.unknowns)
    @test fom.linear[row_of(v), row_of(v)] == 1
    @test fom.linear[row_of(v), row_of(u[1, 1])] == 1
    @test all(symbolically_zero(fom.forcing[row_of(u[i, j])] - sin(t)) for i in 1:2, j in 1:2)
    @test all(symbolically_zero(fom.nonlinear[row_of(u[i, j])] + β * u[i, j]^2) for i in 1:2, j in 1:2)
    @test iszero(fom.nonlinear[row_of(v)])

    fom = ModelOrderReduction.FullOrderModel(
        complete(sys); training_parameters = Dict(Symbolics.unwrap(β) => 5.0)
    )
    @test collect(values(fom.parameter_values)) == [5.0]
end
