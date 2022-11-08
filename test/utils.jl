using Test, ModelOrderReduction
using Symbolics

@variables t w(t) x(t) y(t) z(t)

@testset "linear_terms" begin
    @testset "linear_terms full" begin
        vars = [x, y, z]
        exprs = [3.0x + 4.5y + 6.0
                 2.0z + 3.4w + 7.0 + sin(x)
                 9.8 + x * (1.0 - y)
                 5.6y + 1.3z^2]
        A, c, n = ModelOrderReduction.linear_terms(exprs, vars)
        @test size(A) == (length(exprs), length(vars))
        @test A == [3.0 4.5 0.0
                    0.0 0.0 2.0
                    0.0 0.0 0.0
                    0.0 5.6 0.0]
        @test length(c) == length(exprs)
        @test isequal(c, [6.0, 3.4w + 7.0, 9.8, 0.0])
        @test length(n) == length(exprs)
        @test isequal(n, [0.0, sin(x), x * (1.0 - y), 1.3z^2])
    end

    @testset "linear_terms empty exprs" begin
        vars = [x, y, z]
        exprs = Vector{Num}(undef, 4)
        fill!(exprs, false)
        A, c, n = ModelOrderReduction.linear_terms(exprs, vars)
        @test size(A) == (length(exprs), length(vars))
        @test iszero(A)
        @test length(c) == length(exprs)
        @test iszero(c)
        @test length(n) == length(exprs)
        @test iszero(n)
    end

    @testset "linear_terms diagonal" begin
        vars = [x, y, z]
        exprs = [x, 2y, 3z, 4w]
        A, c, n = ModelOrderReduction.linear_terms(exprs, vars)
        @test size(A) == (length(exprs), length(vars))
        @test A == [1.0 0.0 0.0
                    0.0 2.0 0.0
                    0.0 0.0 3.0
                    0.0 0.0 0.0]
        @test length(c) == length(exprs)
        @test isequal(c, [0.0, 0.0, 0.0, 4w])
        @test length(n) == length(exprs)
        @test iszero(n)
    end

    @testset "linear_terms nonunique vars" begin
        vars = [x, y, y]
        exprs = [3.0x + 4.5y + 6.0
                 2.0z + 3.4w + 7.0 + sin(x)
                 9.8 + x * (1.0 - y)
                 5.6y + 1.3z^2]
        @test_throws ArgumentError ModelOrderReduction.linear_terms(exprs, vars)
    end
end
