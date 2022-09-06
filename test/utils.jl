using Test, ModelOrderReduction
using Symbolics

@testset "linear_terms" begin
    @variables t w(t) x(t) y(t) z(t)
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
