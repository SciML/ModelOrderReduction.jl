using Test
using LinearAlgebra
using ModelOrderReduction

@testset "opinf: recovers linear diagonal dynamics" begin
    Atrue = Diagonal([-1.0, -2.0, -3.0])
    ts = 0:0.05:4
    X = reduce(hcat, [[exp(Atrue[i, i] * t) for i in 1:3] for t in ts])
    Xdot = Matrix(Atrue) * X
    model = opinf(X, Xdot; basis = Matrix{Float64}(I, 3, 3))
    @test size(model.basis) == (3, 3)
    @test model.A ≈ Matrix(Atrue) atol = 1e-8
    @test model.H === nothing
    @test model.B === nothing
    @test model.c === nothing
    xr = X[:, 1]
    @test reduced_dynamics(model, xr) ≈ model.A * xr

    pod_model = opinf(X, Xdot; nmodes = 2)
    @test size(pod_model.basis) == (3, 2)
    @test size(pod_model.A) == (2, 2)
    Xr = pod_model.basis' * X
    Rr = pod_model.basis' * Xdot
    @test norm(Rr - pod_model.A * Xr) / norm(Rr) < 1e-2
end

@testset "opinf: recovers quadratic system with identity basis" begin
    # ẋ₁ = -2 x₁ + x₁ x₂
    # ẋ₂ = -x₂ + x₂²
    Atrue = [-2.0 0.0; 0.0 -1.0]
    # unique monomials [x1^2, x1*x2, x2^2]
    Htrue = [0.0 1.0 0.0; 0.0 0.0 1.0]
    rng_states = [Float64[x1, x2]
                  for x1 in range(-0.5, 0.5; length = 9),
    x2 in range(-0.5, 0.5; length = 9)]
    X = reduce(hcat, vec(rng_states))
    Xdot = similar(X)
    for j in axes(X, 2)
        x = X[:, j]
        Xdot[:, j] = Atrue * x + Htrue * quadratic_monomials(x)
    end
    model = opinf(
        X, Xdot; basis = Matrix{Float64}(I, 2, 2), linear = true, quadratic = true)
    @test model.A ≈ Atrue atol = 1e-10
    @test model.H ≈ Htrue atol = 1e-10
    x = [0.2, -0.3]
    @test reduced_dynamics(model, x) ≈ Atrue * x + Htrue * quadratic_monomials(x)
end

@testset "opinf: forced linear system with inputs" begin
    Atrue = [-1.0 0.0; 0.0 -2.0]
    Btrue = reshape([1.0, 0.5], 2, 1)
    ts = 0:0.02:2
    U = reshape([sin(4t) for t in ts], 1, :)
    # Integrate exactly for diagonal A with forcing: use discrete update from known ODE solution
    # Generate consistent (X, Xdot, U) pairs on a grid of states instead.
    xs = [Float64[x1, x2]
          for x1 in range(-1, 1; length = 7), x2 in range(-1, 1; length = 7)]
    us = range(-1, 1; length = 5)
    cols_x = Vector{Float64}[]
    cols_dx = Vector{Float64}[]
    cols_u = Vector{Float64}[]
    for x in vec(xs), u in us

        push!(cols_x, x)
        push!(cols_dx, Atrue * x + Btrue * [u])
        push!(cols_u, [u])
    end
    X = reduce(hcat, cols_x)
    Xdot = reduce(hcat, cols_dx)
    Umat = reduce(hcat, cols_u)
    model = opinf(X, Xdot; basis = Matrix{Float64}(I, 2, 2), inputs = Umat)
    @test model.A ≈ Atrue atol = 1e-10
    @test model.B ≈ Btrue atol = 1e-10
end

@testset "opinf: vector-of-snapshots API and regularization" begin
    Xcols = [[exp(-0.5t), exp(-1.5t)] for t in 0:0.1:2]
    Xdotcols = [[-0.5 * exp(-0.5t), -1.5 * exp(-1.5t)] for t in 0:0.1:2]
    model = opinf(Xcols, Xdotcols; nmodes = 2, λ = 1e-12)
    @test size(model.A) == (2, 2)
    @test size(model.basis, 2) == 2
end

@testset "opinf: constant term and input evaluation" begin
    Atrue = [-1.0 0.0; 0.0 -2.0]
    ctrue = [0.25, -0.5]
    Btrue = reshape([1.0, 0.5], 2, 1)
    xs = [Float64[x1, x2]
          for x1 in range(-1, 1; length = 6), x2 in range(-1, 1; length = 6)]
    us = range(-1, 1; length = 5)
    cols_x = Vector{Float64}[]
    cols_dx = Vector{Float64}[]
    cols_u = Float64[]
    for x in vec(xs), u in us

        push!(cols_x, x)
        push!(cols_dx, Atrue * x + Btrue * [u] + ctrue)
        push!(cols_u, u)
    end
    X = reduce(hcat, cols_x)
    Xdot = reduce(hcat, cols_dx)
    model = opinf(
        X,
        Xdot;
        basis = Matrix{Float64}(I, 2, 2),
        inputs = cols_u,
        constant = true
    )
    @test model.A ≈ Atrue atol = 1e-10
    @test model.B ≈ Btrue atol = 1e-10
    @test model.c ≈ ctrue atol = 1e-10
    x = [0.2, -0.1]
    u = [0.3]
    @test reduced_dynamics(model, x, u) ≈ Atrue * x + Btrue * u + ctrue
    @test_throws ArgumentError reduced_dynamics(model, x)
end

@testset "opinf: quadratic monomials for r=3" begin
    x = [2.0, 3.0, 5.0]
    @test quadratic_monomials(x) ≈ [4.0, 6.0, 10.0, 9.0, 15.0, 25.0]
end

@testset "opinf: argument checks" begin
    X = rand(3, 5)
    Xdot = rand(3, 4)
    @test_throws DimensionMismatch opinf(X, Xdot; nmodes = 2)
    @test_throws ArgumentError opinf(rand(3, 5), rand(3, 5); nmodes = 2, linear = false)
end
