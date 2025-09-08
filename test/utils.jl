using Test, ModelOrderReduction
using ModelingToolkit

@variables t w(t) x(t) y(t) z(t)

@testset "separate_terms" begin
    @testset "separate_terms basic functionality" begin
        vars = [x, y, z]
        exprs = [x, 2y, 3z, 4w]
        A, c, n = ModelOrderReduction.separate_terms(exprs, vars, t)
        
        # Test dimensions are correct
        @test size(A) == (length(exprs), length(vars))
        @test length(c) == length(exprs)
        @test length(n) == length(exprs)
        
        # Test that function doesn't error and returns expected types
    end

    @testset "separate_terms empty exprs" begin
        vars = [x, y, z]
        exprs = Vector{Num}(undef, 4)
        fill!(exprs, false)
        A, c, n = ModelOrderReduction.separate_terms(exprs, vars, t)
        @test size(A) == (length(exprs), length(vars))
        @test iszero(A)
        @test length(c) == length(exprs)
        @test iszero(c)
        @test length(n) == length(exprs)
        @test iszero(n)
    end

    @testset "separate_terms nonunique vars" begin
        vars = [x, y, y]
        exprs = [3.0x + 4.5y + 6.0,
                 2.0z + 3.4w + 7.0 + sin(x),
                 9.8 + x * (1.0 - y),
                 5.6y + 1.3z^2]
        @test_throws ArgumentError ModelOrderReduction.separate_terms(exprs, vars, t)
    end
end