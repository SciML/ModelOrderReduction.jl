using Test, ModelOrderReduction
using Symbolics, PolyChaos
using SpecialFunctions: beta
import ModelOrderReduction as MOR

supp = (-1, 1)
w(t) = 1 + t
my_meas = Measure("my_meas", w, supp, false, Dict())
deg = 4
Nrec = 2 * deg
my_op = OrthoPoly("my_op", deg, my_meas; Nrec)

@testset "PCE constructor" begin
    @variables t x[1:4] y(t)[1:2]
    degrees = [3, 4, 5, 4]
    Nrec = 2 * minimum(degrees) + 1
    uni_basis = [
        HermiteOrthoPoly(degrees[1]; Nrec),
        Uniform01OrthoPoly(degrees[2]; Nrec),
        Uniform_11OrthoPoly(degrees[3]; Nrec),
        OrthoPoly("my_op", degrees[4], my_meas; Nrec),
    ]
    pce = @test_nowarn PCE(y, [t], x, uni_basis)
    @test isequal(pce.states, Symbolics.scalarize(y))
    @test isequal(pce.parameters, Symbolics.scalarize(x))
    multi_indices_dim = MOR.multi_indices_size(degrees)
    @test size(pce.moments) == (multi_indices_dim, length(y))
    Nrec = 2 * minimum(degrees[1:3]) + 1
    uni_basis = [
        HermiteOrthoPoly(degrees[1]; Nrec),
        Uniform01OrthoPoly(degrees[2]; Nrec),
        Uniform_11OrthoPoly(degrees[3]; Nrec),
    ]
    @test_nowarn PCE(y, [t], x[1:3], uni_basis)
    Nrec = 2 * minimum(degrees[1:2]) + 1
    uni_basis = [HermiteOrthoPoly(degrees[1]; Nrec), HermiteOrthoPoly(degrees[2]; Nrec)]
    @test_nowarn PCE(y, [t], x[1:2], uni_basis)
    uni_basis = [my_op, my_op]
    @test_nowarn PCE(y, [t], x[1:2], uni_basis)
end

@testset "TensorProductOrthoPoly" begin
    ops = [GaussOrthoPoly(1)]
    @test_nowarn MOR.TensorProductOrthoPoly(ops)
    ops = [GaussOrthoPoly(4), GaussOrthoPoly(3)]
    @test_nowarn MOR.TensorProductOrthoPoly(ops)
    ops = [my_op]
    @test_nowarn MOR.TensorProductOrthoPoly(ops)
    ops = [my_op, my_op]
    @test_nowarn MOR.TensorProductOrthoPoly(ops)
    ops = [GaussOrthoPoly(4), Uniform01OrthoPoly(2), LaguerreOrthoPoly(3), my_op]
    tpop = @test_nowarn MOR.TensorProductOrthoPoly(ops)
    @test dim(tpop) == MOR.multi_indices_size([4, 2, 3, 4])
    ops = [GaussOrthoPoly(4; addQuadrature = false), Uniform01OrthoPoly(2)]
    @test_throws InconsistencyError MOR.TensorProductOrthoPoly(ops)
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
        mx = maximum(r)
        @test allunique(view(res, :, i) for i in axes(res, 2))
        for i in axes(res, 2)
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
        tt = Tensor(t, tpop)
        mt = Tensor(t, mop)
        index = Vector{Int}(undef, t)
        for i in 0:(dim(mop) - 1)
            fill!(index, i)
            @test tt.get(index) == mt.get(index)
        end
    end
end

# testing GRevLex capabilities
@testset "PCE: GRevLeX ordering" begin
    @test MOR.grevlex(1, 5) == reshape([5], 1, 1)

    ind = [2 0 0;
           1 1 0;
           0 2 0;
           1 0 1;
           0 1 1;
           0 0 2]
    @test ind == MOR.grevlex(3, 2)

    ind = [1 1 0;
           1 0 1;
           0 1 1;
           0 0 2]
    @test ind == MOR.grevlex(3, 2, [1, 1, 2])

    ind = [1 1 0;
           0 1 1]
    @test ind == MOR.grevlex(3, 2, [0:1, [1], 0:2])

    @test zeros(Int, 0, 3) == MOR.grevlex(3, 0, [0:1, [1], 0:2])

    ind0 = [0 0 0]
    ind1 = [1 0 0;
            0 1 0;
            0 0 1]
    ind2 = [2 0 0;
            1 1 0;
            0 2 0;
            1 0 1;
            0 1 1;
            0 0 2]
    @test vcat(ind0, ind1, ind2) == MOR.grevlex(3, 0:2)
    @test vcat(ind1, ind2) == MOR.grevlex(3, 1:2)
    @test vcat(ind0, ind2) == MOR.grevlex(3, [0, 2])
    @test vcat(ind2, ind0) == MOR.grevlex(3, [2, 0])

    degree_constraints = [0:1, [1], 1:2]
    ind0 = zeros(Int, 0, 3)
    ind1 = zeros(Int, 0, 3)
    ind2 = [0 1 1]
    @test vcat(ind0, ind1, ind2) == MOR.grevlex(3, 0:2, degree_constraints)
    @test vcat(ind1, ind2) == MOR.grevlex(3, 1:2, degree_constraints)
    @test vcat(ind0, ind2) == MOR.grevlex(3, [0, 2], degree_constraints)
    @test vcat(ind2, ind0) == MOR.grevlex(3, [2, 0], degree_constraints)
end

@testset "BasisProduct" begin
    @test MOR.BasisProduct(Int[]) == MOR.BasisProduct(Int[])
    @test MOR.BasisProduct([1, 3]) == MOR.BasisProduct([1, 3])

    @variables x y z
    yv = Symbolics.value(y)
    Ψ₁ = MOR.BasisProduct([4])
    Ψ₂ = MOR.BasisProduct([7])

    expr1 = x * y * z
    dict = Dict(x => Ψ₁, z => Ψ₂)
    expr2 = substitute(expr1, dict)
    expr3 = expand(expr2)
    @test isequal(expr3, yv * MOR.BasisProduct([4, 7]))

    expr4 = (yv + Ψ₁)^2
    expr5 = expand(expr4)
    @test isequal(expr5, yv^2 + 2yv * MOR.BasisProduct([4]) + MOR.BasisProduct([4, 4]))
end
