using Test, ModelOrderReduction
using Symbolics, PolyChaos

@testset "PCE constructor" begin
    x = @variables a b c d
    supp = (-1, 1)
    w(t) = 1 + t
    my_meas = Measure("my_meas", w, supp, false, Dict())
    deg = 4
    my_op = OrthoPoly("my_op", deg, my_meas; Nquad = 200)
    basis = [HermiteOrthoPoly(3), Uniform01OrthoPoly(4), Uniform_11OrthoPoly(5), my_op]
    ind = rand(Int, 5, 4)
    @test_nowarn PCE(x, basis, ind)
end
