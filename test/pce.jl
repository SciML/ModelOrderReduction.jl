using Test, ModelOrderReduction
using Symbolics, PolyChaos
using SpecialFunctions: beta
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

@testset "multi_indices_size" begin
    @test MOR.multi_indices_size([0]) == 1
    @test MOR.multi_indices_size([1]) == 2
    @test MOR.multi_indices_size([5]) == 6
    @test MOR.multi_indices_size([2, 2]) == 6
    @test MOR.multi_indices_size([4, 2]) == 12
    @test MOR.multi_indices_size([2, 2, 2]) == 10
end

@testset "grlex" begin
    @test MOR.grlex([0]) == zeros(Int, 1, 1)
    @test MOR.grlex([1]) == [0 1]
    @test MOR.grlex([5]) == [0 1 2 3 4 5]
    @test MOR.grlex([2, 2]) == [0 1 0 2 1 0
                                0 0 1 0 1 2]
    @test MOR.grlex([4, 2]) == [0 1 0 2 1 0 3 2 1 4 3 2
                                0 0 1 0 1 2 0 1 2 0 1 2]
    @test MOR.grlex([2, 2, 2]) ==
          [0 1 0 0 2 1 1 0 0 0
           0 0 1 0 0 1 0 2 1 0
           0 0 0 1 0 0 1 0 1 2]
    @test MOR.grlex([3, 4, 2]) ==
          [0 1 0 0 2 1 1 0 0 0 3 2 2 1 1 1 0 0 0 3 3 2 2 2 1 1 1 0 0 0
           0 0 1 0 0 1 0 2 1 0 0 1 0 2 1 0 3 2 1 1 0 2 1 0 3 2 1 4 3 2
           0 0 0 1 0 0 1 0 1 2 0 0 1 0 1 2 0 1 2 0 1 0 1 2 0 1 2 0 1 2]

    function check_grlex(r::AbstractVector{<:Integer})
        res = MOR.grlex(r)
        col = size(res, 2)
        mx = maximum(r)
        @test allunique(view(res, :, i) for i in 1:col)
        for i in 1:col
            degree = @view res[:, i]
            @test sum(degree) ≤ mx
            @test all(degree .≤ r)
        end
    end
    check_grlex([4, 1, 5])
    check_grlex([3, 2, 2])
    check_grlex([3, 2, 1, 2])
    check_grlex([2, 5, 3, 4, 2])
end

@testset "Tensor" begin
    deg, n = 4, 20
    s_α, s_β = 2.1, 3.2
    beta_op = Beta01OrthoPoly(deg, s_α, s_β; Nrec = n, addQuadrature = true)
    supp = (0, 1)
    w(t) = (t^(s_α - 1) * (1 - t)^(s_β - 1) / beta(s_α, s_β))
    my_meas = Measure("my_meas", w, supp, false)
    my_opq = OrthoPoly("my_op", deg, my_meas; Nrec = n, addQuadrature = true)

    ops = [beta_op, my_opq]
    tpop = MOR.TensorProductOrthoPoly(ops)
    mop = MultiOrthoPoly(ops, deg)

    for t in (2, 3)
        tt = MOR.Tensor2(t, tpop)
        mt = Tensor(t, mop)
        index = Vector{Int}(undef, t)
        for i in 0:(dim(mop) - 1)
            fill!(index, i)
            @test tt.get(index) == mt.get(index)
        end
    end
end