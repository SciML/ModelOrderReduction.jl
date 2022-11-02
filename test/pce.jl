using Test, ModelOrderReduction
using Symbolics, PolyChaos
import ModelOrderReduction as MOR

supp = (-1, 1)
w(t) = 1 + t
my_meas = Measure("my_meas", w, supp, false, Dict())
deg = 4
my_op = OrthoPoly("my_op", deg, my_meas; Nquad = 200)

@testset "PCE constructor" begin
    x = @variables a b c d
    basis = [HermiteOrthoPoly(3), Uniform01OrthoPoly(4), Uniform_11OrthoPoly(5), my_op]
    ind = rand(Int, 5, 4)
    @test_nowarn PCE(x, basis, ind)
end

@testset "TensorProductOrthoPoly" begin
    ops = [GaussOrthoPoly(4), Uniform01OrthoPoly(2), LaguerreOrthoPoly(3), my_op]
    @test_nowarn MOR.TensorProductOrthoPoly(ops)
end
