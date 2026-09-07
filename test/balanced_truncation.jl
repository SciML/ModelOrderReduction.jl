using Test
using LinearAlgebra
using ModelOrderReduction

@testset "baltrunc: diagonal decay recovers dominant mode" begin
    A = Diagonal([-1.0, -10.0, -100.0])
    B = reshape([1.0, 0.1, 0.01], 3, 1)
    C = [1.0 0.1 0.01]
    bt = baltrunc(Matrix(A), B, C; n = 1)
    @test size(bt.A) == (1, 1)
    @test length(bt.hsv) == 1
    @test bt.hsv[1] > 0
    @test size(bt.Tr) == (3, 1)
    @test size(bt.S) == (1, 3)
    @test bt.S * bt.Tr ≈ I(1) atol = 1e-10
    # Dominant pole near -1
    @test only(eigvals(bt.A)) ≈ -1.0 atol = 5e-2
end

@testset "baltrunc: order selection by atol/rtol" begin
    A = Diagonal([-1.0, -2.0, -50.0])
    B = Matrix{Float64}(I, 3, 3)
    C = Matrix{Float64}(I, 3, 3)
    bt = baltrunc(Matrix(A), B, C; atol = 1e-8, rtol = 1e-2)
    @test 1 <= size(bt.A, 1) <= 3
    @test issorted(bt.hsv; rev = true)
end

@testset "baltrunc: residualization matches DC gain better than truncation" begin
    A = [-1.0 0.0; 0.0 -10.0]
    B = reshape([1.0, 1.0], 2, 1)
    C = [1.0 1.0]
    D = zeros(1, 1)
    full_dc = only(-C * (A \ B) + D)
    trunc = baltrunc(A, B, C, D; n = 1, residual = false)
    resid = baltrunc(A, B, C, D; n = 1, residual = true)
    trunc_dc = only(-trunc.C * (trunc.A \ trunc.B) + trunc.D)
    resid_dc = only(-resid.C * (resid.A \ resid.B) + resid.D)
    @test abs(resid_dc - full_dc) <= abs(trunc_dc - full_dc) + 1e-12
    @test resid_dc ≈ full_dc atol = 1e-10
end

@testset "baltrunc: dimension checks" begin
    A = [-1.0 0.0; 0.0 -2.0]
    B = reshape([1.0, 1.0], 2, 1)
    C = [1.0 1.0]
    @test_throws ArgumentError baltrunc(A, B, C; n = 0)
    @test_throws ArgumentError baltrunc([1.0 2.0], B, C)
    @test_throws ArgumentError baltrunc(A, reshape([1.0], 1, 1), C)
    @test_throws ArgumentError baltrunc(Diagonal([0.5, -2.0]), B, C; n = 1)
end
