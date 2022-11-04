using Test, ModelOrderReduction
using Symbolics, PolyChaos
import ModelOrderReduction: TensorProductOrthoPoly, multi_indices_size

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
    @test_nowarn TensorProductOrthoPoly(ops)
end

@testset "multi_indices_size" begin
    @test multi_indices_size([0]) == 1
    @test multi_indices_size([1]) == 2
    @test multi_indices_size([5]) == 6
    @test multi_indices_size([2, 2]) == 6
    @test multi_indices_size([4, 2]) == 12
    @test multi_indices_size([2, 2, 2]) == 10
end
