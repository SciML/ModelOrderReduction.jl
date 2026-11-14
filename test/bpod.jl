using Test
using LinearAlgebra
using SparseArrays
using ModelOrderReduction

const A_PLANT = [
    -1.0 20.0 0.0 0.0
    0.0 -2.0 15.0 0.0
    0.0 0.0 -3.0 10.0
    0.0 0.0 0.0 -6.0
]
const B_PLANT = [
    1.0 0.0
    0.4 1.0
    0.0 0.5
    0.2 -0.3
]
const C_PLANT = [
    1.0 0.2 0.0 0.4
    0.0 1.0 0.3 -0.2
]
const D_PLANT = [
    0.15 -0.02
    0.01 0.08
]

function psd_sqrt(W::AbstractMatrix{T}) where {T <: AbstractFloat}
    F = eigen(Symmetric(Matrix{T}(W)))
    λ = F.values
    λmax = maximum(abs, λ)
    thresh = eps(T) * λmax
    kept = findall(≥(thresh), λ)
    isempty(kept) && return zeros(T, size(W, 1), 0)
    return F.vectors[:, kept] * Diagonal(sqrt.(λ[kept]))
end

function exact_factors(A, B, C)
    Pc = lyap(Matrix(A), Matrix(B) * Matrix(B)')
    Po = lyap(Matrix(A)', Matrix(C)' * Matrix(C))
    return psd_sqrt(Pc), psd_sqrt(Po)
end

function relative_error(got, reference)
    return norm(got - reference) / norm(reference)
end

function uniform_nodes(horizon::Real, dt::Real)
    nsteps = round(Int, horizon / dt)
    abs(nsteps * dt - horizon) <= 100 * eps(float(horizon)) * max(horizon, one(float(horizon))) ||
        throw(ArgumentError("dt = $dt does not divide the horizon T = $horizon"))
    return collect(range(0.0, float(horizon); length = nsteps + 1))
end

function trapezoidal_weights(t::AbstractVector{<:Real})
    nt = length(t)
    nt >= 2 || throw(ArgumentError("trapezoidal rule needs at least two nodes"))
    all(>(0), diff(t)) || throw(ArgumentError("time nodes must be strictly increasing"))
    w = similar(float.(t))
    w[1] = (t[2] - t[1]) / 2
    for k in 2:(nt - 1)
        w[k] = (t[k + 1] - t[k - 1]) / 2
    end
    w[end] = (t[end] - t[end - 1]) / 2
    return w
end

function impulse_factor(generator::AbstractMatrix, columns::AbstractMatrix, t, w)
    n, m = size(columns)
    nt = length(t)
    length(w) == nt || throw(ArgumentError("weights and nodes must have the same length"))
    X = zeros(eltype(generator), n, m * nt)
    for k in eachindex(t)
        response = exp(generator * t[k])
        scale = sqrt(w[k])
        for j in 1:m
            X[:, (j - 1) * nt + k] = scale * (response * columns[:, j])
        end
    end
    return X
end

function weighted_impulses(A, B, C, t)
    w = trapezoidal_weights(t)
    return impulse_factor(A, B, t, w), impulse_factor(A', C', t, w)
end

function component_relative_error(got, reference)
    return [abs(got[i] - reference[i]) / reference[i] for i in eachindex(reference)]
end

function transfer(A, B, C, D, ω)
    n = size(A, 1)
    return C * (((im * ω) * I(n) - A) \ B) + D
end

function sampled_peak_difference(A, B, C, D, Ar, Br, Cr, Dr, frequencies)
    return maximum(frequencies) do ω
        opnorm(transfer(A, B, C, D, ω) - transfer(Ar, Br, Cr, Dr, ω))
    end
end

function h2_difference(A, B, C, Ar, Br, Cr)
    n = size(A, 1)
    r = size(Ar, 1)
    Ae = [A zeros(n, r); zeros(r, n) Ar]
    maximum(real, eigvals(Ae)) < 0 ||
        error("the difference system must be Hurwitz before its Lyapunov solution is an H2 error")
    Be = [B; Br]
    Ce = [C -Cr]
    P = lyap(Ae, Be * Be')
    return sqrt(abs(tr(Ce * P * Ce')))
end

function align_signature(reference::BalancedTruncation, model::BalancedTruncation)
    r = size(model.Tr, 2)
    signs = ones(eltype(model.A), r)
    for j in 1:r
        if dot(reference.Tr[:, j], model.Tr[:, j]) < 0
            signs[j] = -1
        end
    end
    E = Diagonal(signs)
    return BalancedTruncation(
        E * model.A * E,
        E * model.B,
        model.C * E,
        model.D,
        copy(model.hsv),
        model.Tr * E,
        E * model.S
    )
end

@testset "bpod: exact factors agree with baltrunc" begin
    X, Y = exact_factors(A_PLANT, B_PLANT, C_PLANT)
    for r in (2, 4)
        reference = baltrunc(A_PLANT, B_PLANT, C_PLANT, D_PLANT; n = r, atol = 0.0, rtol = 0.0)
        model = bpod(A_PLANT, B_PLANT, C_PLANT, X, Y, D_PLANT; n = r, atol = 0.0, rtol = 0.0)
        aligned = align_signature(reference, model)
        @test model isa BalancedTruncation
        @test size(model.A) == (r, r)
        @test size(model.Tr) == (4, r)
        @test size(model.S) == (r, 4)
        @test length(model.hsv) == r
        @test model.hsv ≈ reference.hsv rtol = 1.0e-8 atol = 1.0e-8
        @test relative_error(aligned.A, reference.A) < 1.0e-8
        @test relative_error(aligned.B, reference.B) < 1.0e-8
        @test relative_error(aligned.C, reference.C) < 1.0e-8
        @test model.D == reference.D
        @test model.D == D_PLANT
        @test norm(model.S * model.Tr - I(r)) < 1.0e-10
    end
end

@testset "bpod: mixed Float32 and Float64 promote to Float64" begin
    X, Y = exact_factors(A_PLANT, B_PLANT, C_PLANT)
    A32 = Float32.(A_PLANT)
    X32 = Float32.(X)
    promoted = bpod(
        Float64.(A32), B_PLANT, C_PLANT, Float64.(X32), Y, D_PLANT;
        n = 2, atol = 0.0, rtol = 0.0
    )
    mixed = bpod(A32, B_PLANT, C_PLANT, X32, Y, D_PLANT; n = 2, atol = 0.0, rtol = 0.0)
    aligned = align_signature(promoted, mixed)
    @test eltype(mixed.A) == Float64
    @test eltype(mixed.hsv) == Float64
    @test relative_error(aligned.A, promoted.A) < 1.0e-10
    @test relative_error(aligned.B, promoted.B) < 1.0e-10
    @test relative_error(aligned.C, promoted.C) < 1.0e-10
    @test aligned.hsv ≈ promoted.hsv rtol = 1.0e-10 atol = 1.0e-10
    @test mixed.D == promoted.D
end

@testset "bpod: invalid input" begin
    X, Y = exact_factors(A_PLANT, B_PLANT, C_PLANT)
    @test_throws ArgumentError bpod([1.0 2.0], B_PLANT, C_PLANT, X, Y)
    @test_throws ArgumentError bpod(A_PLANT, ones(1, 1), C_PLANT, X, Y)
    @test_throws ArgumentError bpod(A_PLANT, B_PLANT, ones(2, 3), X, Y)
    @test_throws ArgumentError bpod(A_PLANT, B_PLANT, C_PLANT, X, Y, ones(1, 1))
    @test_throws ArgumentError bpod(A_PLANT, B_PLANT, C_PLANT, X[1:3, :], Y)
    @test_throws ArgumentError bpod(A_PLANT, B_PLANT, C_PLANT, X, Y[1:3, :])
    @test_throws ArgumentError bpod(A_PLANT, B_PLANT, C_PLANT, X[:, 1:0], Y)
    @test_throws ArgumentError bpod(A_PLANT, B_PLANT, C_PLANT, X, Y[:, 1:0])
    @test_throws ArgumentError bpod(complex.(A_PLANT), B_PLANT, C_PLANT, X, Y)
    @test_throws ArgumentError bpod(A_PLANT, B_PLANT, C_PLANT, X, Y; n = 0)
    @test_throws ArgumentError bpod(A_PLANT, B_PLANT, C_PLANT, X, Y; n = -1)
    @test_throws ArgumentError bpod(A_PLANT, B_PLANT, C_PLANT, X, Y; n = 1.5)
    @test_throws ArgumentError bpod(A_PLANT, B_PLANT, C_PLANT, X, Y; n = 5)
    @test_throws ArgumentError bpod(A_PLANT, B_PLANT, C_PLANT, zeros(4, 2), zeros(4, 2); n = 1)
    @test_throws ArgumentError bpod(A_PLANT, B_PLANT, C_PLANT, X, Y; n = 1, atol = 1.0e6)

    Xnan = copy(X)
    Xnan[1, 1] = NaN
    @test_throws "X contains non-finite" bpod(A_PLANT, B_PLANT, C_PLANT, Xnan, Y)
    Yinf = copy(Y)
    Yinf[1, 1] = Inf
    @test_throws "Y contains non-finite" bpod(A_PLANT, B_PLANT, C_PLANT, X, Yinf)
end

@testset "bpod: sparse Float32 operators stay sparse and match dense Float64" begin
    X, Y = exact_factors(A_PLANT, B_PLANT, C_PLANT)
    A32 = SparseMatrixCSC{Float32, Int32}(sparse(Float32.(A_PLANT)))
    B32 = sparse(Float32.(B_PLANT))
    C32 = sparse(Float32.(C_PLANT))
    converted = ModelOrderReduction._coerce_operator(A32, Float64)
    @test converted isa SparseMatrixCSC{Float64}
    @test nnz(converted) == nnz(A32)
    @test converted == Float64.(A32)

    dense = bpod(
        Matrix{Float64}(A32), Matrix{Float64}(B32), Matrix{Float64}(C32), X, Y, D_PLANT;
        n = 2, atol = 0.0, rtol = 0.0
    )
    sparse_model = bpod(A32, B32, C32, X, Y, D_PLANT; n = 2, atol = 0.0, rtol = 0.0)
    aligned = align_signature(dense, sparse_model)
    @test eltype(sparse_model.A) == Float64
    @test relative_error(aligned.A, dense.A) < 1.0e-10
    @test relative_error(aligned.B, dense.B) < 1.0e-10
    @test relative_error(aligned.C, dense.C) < 1.0e-10
    @test aligned.hsv ≈ dense.hsv rtol = 1.0e-10 atol = 1.0e-10

    stored = sparse(Float64[1.0 0.0; 0.0 -2.0])
    @test ModelOrderReduction._assert_finite("A", stored) === nothing
    stored[1, 1] = NaN
    @test_throws "A contains non-finite" ModelOrderReduction._assert_finite("A", stored)
end

@testset "bpod: order selection matches baltrunc" begin
    A = Matrix(Diagonal([-1.0, -2.0, -50.0]))
    B = Matrix{Float64}(I, 3, 3)
    C = Matrix{Float64}(I, 3, 3)
    X, Y = exact_factors(A, B, C)
    # Margins stay away from the cutoffs. The Hankel values are about 0.5, 0.25, and 0.01.
    for rtol in (0.0, 1.0e-2, 0.1)
        reference = baltrunc(A, B, C; n = nothing, atol = 0.0, rtol = rtol)
        model = bpod(A, B, C, X, Y; n = nothing, atol = 0.0, rtol = rtol)
        aligned = align_signature(reference, model)
        @test size(model.A, 1) == size(reference.A, 1)
        @test issorted(model.hsv; rev = true)
        @test model.hsv ≈ reference.hsv rtol = 1.0e-8 atol = 1.0e-8
        @test relative_error(aligned.A, reference.A) < 1.0e-8
        @test relative_error(aligned.B, reference.B) < 1.0e-8
        @test relative_error(aligned.C, reference.C) < 1.0e-8
        @test model.D == reference.D == zeros(3, 3)
    end
    @test_throws ArgumentError baltrunc(A, B, C; n = nothing, atol = 0.0, rtol = 2.0)
    @test_throws "no Hankel singular values satisfy" bpod(
        A, B, C, X, Y; n = nothing, atol = 0.0, rtol = 2.0
    )

    X, Y = exact_factors(A_PLANT, B_PLANT, C_PLANT)
    reference = baltrunc(A_PLANT, B_PLANT, C_PLANT, D_PLANT)
    model = bpod(A_PLANT, B_PLANT, C_PLANT, X, Y, D_PLANT)
    aligned = align_signature(reference, model)
    @test size(model.A, 1) == size(reference.A, 1) == 4
    @test model.hsv ≈ reference.hsv rtol = 1.0e-8 atol = 1.0e-8
    @test relative_error(aligned.A, reference.A) < 1.0e-8
    @test model.D == reference.D == D_PLANT
end

@testset "bpod: rank-deficient factors reject an oversized order" begin
    X, Y = exact_factors(A_PLANT, B_PLANT, C_PLANT)
    X1 = repeat(X[:, 1:1], 1, 2)
    Y1 = repeat(Y[:, 1:1], 1, 2)
    @test_throws "exceeds the 1 positive" bpod(
        A_PLANT, B_PLANT, C_PLANT, X1, Y1; n = 2, atol = 0.0, rtol = 0.0
    )
    model = bpod(A_PLANT, B_PLANT, C_PLANT, X1, Y1; n = nothing, atol = 0.0, rtol = 0.0)
    @test size(model.A) == (1, 1)
    @test length(model.hsv) == 1
    @test only(model.hsv) > 0
    @test norm(model.S * model.Tr - I(1)) < 1.0e-10
end

@testset "bpod: vector input and omitted feedthrough" begin
    X, Y = exact_factors(A_PLANT, B_PLANT, C_PLANT)
    column = B_PLANT[:, 1]
    feedthrough = D_PLANT[:, 1:1]
    from_vector = bpod(
        A_PLANT, column, C_PLANT, X, Y, feedthrough; n = 2, atol = 0.0, rtol = 0.0
    )
    from_matrix = bpod(
        A_PLANT, reshape(column, :, 1), C_PLANT, X, Y, feedthrough;
        n = 2, atol = 0.0, rtol = 0.0
    )
    @test from_vector.A == from_matrix.A
    @test from_vector.B == from_matrix.B
    @test from_vector.C == from_matrix.C
    @test from_vector.D == from_matrix.D == feedthrough
    @test from_vector.hsv == from_matrix.hsv
    @test size(from_vector.B) == (2, 1)

    omitted = bpod(A_PLANT, B_PLANT, C_PLANT, X, Y; n = 2, atol = 0.0, rtol = 0.0)
    explicit = bpod(
        A_PLANT, B_PLANT, C_PLANT, X, Y, zeros(2, 2); n = 2, atol = 0.0, rtol = 0.0
    )
    reference = baltrunc(A_PLANT, B_PLANT, C_PLANT; n = 2, atol = 0.0, rtol = 0.0)
    @test omitted.D == explicit.D == reference.D == zeros(2, 2)
    @test omitted.A == explicit.A
    @test omitted.B == explicit.B
    @test omitted.C == explicit.C
    @test omitted.hsv == explicit.hsv
    single = bpod(A_PLANT, column, C_PLANT, X, Y; n = 1, atol = 0.0, rtol = 0.0)
    @test single.D == zeros(2, 1)
    @test size(single.B) == (1, 1)
end

@testset "bpod: trapezoidal impulse snapshots" begin
    # Composite trapezoidal weights integrate 3 + 2t exactly over [0, 1] (the
    # integral is 4) and sum to the length of the interval.
    irregular = [0.0, 0.2, 0.5, 0.9, 1.0]
    weights = trapezoidal_weights(irregular)
    @test abs(sum(weights .* (3 .+ 2 .* irregular)) - 4.0) < 1.0e-12
    @test abs(sum(weights) - 1.0) < 1.0e-12

    classical = baltrunc(A_PLANT, B_PLANT, C_PLANT, D_PLANT; n = 4, atol = 0.0, rtol = 0.0)
    classical_r2 = baltrunc(A_PLANT, B_PLANT, C_PLANT, D_PLANT; n = 2, atol = 0.0, rtol = 0.0)
    coarse_t = uniform_nodes(2.0, 1.0)
    Xc, Yc = weighted_impulses(A_PLANT, B_PLANT, C_PLANT, coarse_t)
    coarse = bpod(A_PLANT, B_PLANT, C_PLANT, Xc, Yc, D_PLANT; n = 4, atol = 0.0, rtol = 0.0)
    coarse_r2 = bpod(A_PLANT, B_PLANT, C_PLANT, Xc, Yc, D_PLANT; n = 2, atol = 0.0, rtol = 0.0)
    coarse_hsv_error = maximum(component_relative_error(coarse.hsv, classical.hsv))
    coarse_r2_h2 = h2_difference(A_PLANT, B_PLANT, C_PLANT, coarse_r2.A, coarse_r2.B, coarse_r2.C)
    classical_r2_h2 = h2_difference(
        A_PLANT, B_PLANT, C_PLANT, classical_r2.A, classical_r2.B, classical_r2.C
    )
    # On this mesh the projectors stay biorthogonal to about 1e-14, while the
    # largest relative Hankel error is 0.343 and the order-2 H2 errors are 4.340
    # versus 2.548. The inequalities below are those recorded tolerances.
    @test opnorm(coarse.S * coarse.Tr - I(4)) < 1.0e-8
    @test opnorm(coarse_r2.S * coarse_r2.Tr - I(2)) < 1.0e-8
    @test coarse_hsv_error > 0.2
    @test abs(coarse_r2_h2 - classical_r2_h2) / classical_r2_h2 > 0.2

    # T = 16 and Δt = 1/16 has a negligible controllability tail. A reference
    # quadrature table gives relative Hankel errors 1.879e-4 and 1.817e-3 for the
    # first two values. Order-2 H2 errors at Δt = 1/128 differed by about 1.4e-5.
    # Eightfold coarser steps scale a second-order error by 64, to about 9e-4.
    # Each assertion allows about a factor of five over that evidence.
    resolved_t = uniform_nodes(16.0, 1 / 16)
    Xr, Yr = weighted_impulses(A_PLANT, B_PLANT, C_PLANT, resolved_t)
    resolved = bpod(A_PLANT, B_PLANT, C_PLANT, Xr, Yr, D_PLANT; n = 4, atol = 0.0, rtol = 0.0)
    resolved_r2 = bpod(A_PLANT, B_PLANT, C_PLANT, Xr, Yr, D_PLANT; n = 2, atol = 0.0, rtol = 0.0)
    hsv_error = component_relative_error(resolved.hsv, classical.hsv)
    resolved_r2_h2 = h2_difference(
        A_PLANT, B_PLANT, C_PLANT, resolved_r2.A, resolved_r2.B, resolved_r2.C
    )
    @test hsv_error[1] < 1.0e-3
    @test hsv_error[2] < 1.0e-2
    @test abs(resolved_r2_h2 - classical_r2_h2) / classical_r2_h2 < 5.0e-3
    @test abs(resolved_r2_h2 - classical_r2_h2) < abs(coarse_r2_h2 - classical_r2_h2)
    # A reference sampled peak of 3.029e-4 between the order-2 frequency responses
    # at Δt = 1/128, scaled by 64, is about 0.019. The maximum is over five
    # frequencies, not a certified H∞ norm, and the tolerance allows about a
    # factor of five.
    frequencies = (0.1, 0.7, 1.0, 1.3, 10.0)
    frequency_gap = sampled_peak_difference(
        resolved_r2.A, resolved_r2.B, resolved_r2.C, resolved_r2.D,
        classical_r2.A, classical_r2.B, classical_r2.C, classical_r2.D,
        frequencies
    )
    @test frequency_gap < 0.1
end

function agrees_with_dense(Aop, Bop, Cop, X, Y)
    reference = bpod(
        Matrix(Aop), Matrix(Bop), Matrix(Cop), X, Y; n = 2, atol = 0.0, rtol = 0.0
    )
    model = bpod(Aop, Bop, Cop, X, Y; n = 2, atol = 0.0, rtol = 0.0)
    aligned = align_signature(reference, model)
    @test relative_error(aligned.A, reference.A) < 1.0e-10
    @test relative_error(aligned.B, reference.B) < 1.0e-10
    @test relative_error(aligned.C, reference.C) < 1.0e-10
    @test aligned.hsv ≈ reference.hsv rtol = 1.0e-10 atol = 1.0e-10
    return nothing
end

@testset "bpod: structured and sparse-adjoint operators match dense references" begin
    Ad = Diagonal([-1.0, -2.0, -3.0, -4.0])
    Bd = Diagonal([1.0, 0.4, 0.2, 0.1])
    Cd = Diagonal([1.0, 0.5, 0.3, 0.2])
    @test ModelOrderReduction._coerce_operator(Ad, Float64) === Ad
    Xd, Yd = exact_factors(Matrix(Ad), Matrix(Bd), Matrix(Cd))
    agrees_with_dense(Ad, Bd, Cd, Xd, Yd)
    Ad32 = Diagonal(Float32.(diag(Ad)))
    promoted_diag = ModelOrderReduction._coerce_operator(Ad32, Float64)
    @test promoted_diag isa Diagonal{Float64}
    @test promoted_diag.diag == diag(Ad)
    agrees_with_dense(Ad32, Bd, Cd, Xd, Yd)
    @test_throws "A contains non-finite" ModelOrderReduction._assert_finite(
        "A", Diagonal([1.0, NaN, -3.0])
    )

    At = Tridiagonal(fill(0.2, 3), [-1.0, -1.5, -2.0, -2.5], fill(0.3, 3))
    Bt = [1.0, 0.0, 0.0, 0.0]
    Ct = [0.0 0.0 0.0 1.0]
    @test ModelOrderReduction._coerce_operator(At, Float64) === At
    Xt, Yt = exact_factors(Matrix(At), reshape(Bt, :, 1), Ct)
    agrees_with_dense(At, reshape(Bt, :, 1), Ct, Xt, Yt)
    At32 = Tridiagonal(fill(Float32(0.2), 3), Float32[-1.0, -1.5, -2.0, -2.5], fill(Float32(0.3), 3))
    promoted_tri = ModelOrderReduction._coerce_operator(At32, Float64)
    @test promoted_tri isa Tridiagonal{Float64}
    @test promoted_tri.d == Float64.(At32.d)
    @test promoted_tri.dl == Float64.(At32.dl)
    @test promoted_tri.du == Float64.(At32.du)
    agrees_with_dense(At32, reshape(Bt, :, 1), Ct, Xt, Yt)
    @test_throws "A contains non-finite" ModelOrderReduction._assert_finite(
        "A", Tridiagonal([NaN], [-1.0, -2.0], [0.1])
    )

    Ab32 = Bidiagonal(Float32[-1.0, -2.0, -3.0, -4.0], fill(Float32(0.2), 3), :L)
    promoted_bi = ModelOrderReduction._coerce_operator(Ab32, Float64)
    @test promoted_bi isa Bidiagonal{Float64}
    @test promoted_bi.uplo == Ab32.uplo
    @test promoted_bi.dv == Float64.(Ab32.dv)
    @test promoted_bi.ev == Float64.(Ab32.ev)
    Xb, Yb = exact_factors(Matrix(promoted_bi), reshape(Bt, :, 1), Ct)
    agrees_with_dense(Ab32, reshape(Bt, :, 1), Ct, Xb, Yb)

    As32 = SymTridiagonal(Float32[-1.0, -2.0, -3.0, -4.0], fill(Float32(0.2), 3))
    promoted_sym = ModelOrderReduction._coerce_operator(As32, Float64)
    @test promoted_sym isa SymTridiagonal{Float64}
    @test promoted_sym.dv == Float64.(As32.dv)
    @test promoted_sym.ev == Float64.(As32.ev)
    Xs, Ys = exact_factors(Matrix(promoted_sym), reshape(Bt, :, 1), Ct)
    agrees_with_dense(As32, reshape(Bt, :, 1), Ct, Xs, Ys)

    Aadj = adjoint(sparse(A_PLANT'))
    Badj = adjoint(sparse(B_PLANT'))
    Cadj = adjoint(sparse(C_PLANT'))
    @test ModelOrderReduction._coerce_operator(Aadj, Float64) === Aadj
    @test Matrix(Aadj) == A_PLANT
    Xp, Yp = exact_factors(A_PLANT, B_PLANT, C_PLANT)
    agrees_with_dense(Aadj, Badj, Cadj, Xp, Yp)
    Aadj32 = adjoint(sparse(Float32.(A_PLANT')))
    promoted_adj = ModelOrderReduction._coerce_operator(Aadj32, Float64)
    @test promoted_adj isa Adjoint{Float64, <:SparseMatrixCSC{Float64, Int}}
    @test nnz(parent(promoted_adj)) == nnz(parent(Aadj32))
    @test Matrix(promoted_adj) == Float64.(Matrix(Aadj32))
    agrees_with_dense(
        Aadj32, adjoint(sparse(Float32.(B_PLANT'))), adjoint(sparse(Float32.(C_PLANT'))), Xp, Yp
    )
    poisoned = sparse(Float64[1.0 0.0; 0.0 -1.0])
    poisoned[2, 1] = NaN
    @test_throws "A contains non-finite" ModelOrderReduction._assert_finite("A", adjoint(poisoned))
    @test_throws "A contains non-finite" ModelOrderReduction._assert_finite("A", transpose(poisoned))
end

@testset "bpod: Cholesky Gramian factors balance and preserve the full-order map" begin
    # These factors come from cholesky, not from the eigenvalue factor helper.
    Pc = lyap(A_PLANT, B_PLANT * B_PLANT')
    Po = lyap(A_PLANT', C_PLANT' * C_PLANT)
    X = Matrix(cholesky(Hermitian(Pc)).L)
    Y = Matrix(cholesky(Hermitian(Po)).L)
    model = bpod(A_PLANT, B_PLANT, C_PLANT, X, Y, D_PLANT; n = 4, atol = 0.0, rtol = 0.0)
    @test model.S * Pc * model.S' ≈ Diagonal(model.hsv) rtol = 1.0e-8 atol = 1.0e-8
    @test model.Tr' * Po * model.Tr ≈ Diagonal(model.hsv) rtol = 1.0e-8 atol = 1.0e-8
    frequencies = (0.1, 1.0, 10.0)
    @test sampled_peak_difference(
        A_PLANT, B_PLANT, C_PLANT, D_PLANT,
        model.A, model.B, model.C, model.D, frequencies
    ) < 1.0e-8
end

@testset "bpod: unequal snapshot counts, Float32, repeated values, and integers" begin
    b = B_PLANT[:, 1:1]
    tX = uniform_nodes(1.0, 0.5)
    tY = uniform_nodes(1.0, 0.25)
    X = impulse_factor(A_PLANT, b, tX, trapezoidal_weights(tX))
    Y = impulse_factor(A_PLANT', C_PLANT', tY, trapezoidal_weights(tY))
    @test size(X, 2) == 3
    @test size(Y, 2) == 10
    unequal = bpod(
        A_PLANT, b, C_PLANT, X, Y, D_PLANT[:, 1:1]; n = 2, atol = 0.0, rtol = 0.0
    )
    @test size(unequal.B) == (2, 1)
    @test size(unequal.C) == (2, 2)
    @test size(unequal.D) == (2, 1)
    @test opnorm(unequal.S * unequal.Tr - I(2)) < 1.0e-8

    Xf, Yf = exact_factors(A_PLANT, B_PLANT, C_PLANT)
    narrow = bpod(
        Float32.(A_PLANT), Float32.(B_PLANT), Float32.(C_PLANT),
        Float32.(Xf), Float32.(Yf), Float32.(D_PLANT); n = 2, atol = 0.0, rtol = 0.0
    )
    wide = bpod(
        Float64.(Float32.(A_PLANT)), Float64.(Float32.(B_PLANT)), Float64.(Float32.(C_PLANT)),
        Float64.(Float32.(Xf)), Float64.(Float32.(Yf)), Float64.(Float32.(D_PLANT));
        n = 2, atol = 0.0, rtol = 0.0
    )
    aligned = align_signature(wide, narrow)
    @test eltype(narrow.A) == Float32
    @test eltype(narrow.hsv) == Float32
    @test relative_error(aligned.A, wide.A) < 1.0e-5
    @test aligned.hsv ≈ wide.hsv rtol = 1.0e-5 atol = 1.0e-5

    Ar = Matrix(-1.0I, 2, 2)
    Br = Matrix{Float64}(I, 2, 2)
    Pc = lyap(Ar, Br * Br')
    factor = Matrix(cholesky(Hermitian(Pc)).L)
    repeated = bpod(Ar, Br, Br, factor, factor; n = 2, atol = 0.0, rtol = 0.0)
    @test repeated.hsv ≈ [0.5, 0.5] atol = 1.0e-10
    @test repeated.S * Pc * repeated.S' ≈ Diagonal(repeated.hsv) atol = 1.0e-8
    @test opnorm(repeated.S * repeated.Tr - I(2)) < 1.0e-10

    integer = bpod([-1 0; 0 -2], [1, 0], [1 0], [1 0; 0 1], [1 0; 0 1]; n = 1)
    @test eltype(integer.A) == Float64
    @test eltype(integer.hsv) == Float64
end

@testset "bpod: a short horizon can destabilize a Hurwitz system" begin
    # Two nodes on [0, 0.25]. A reference evaluation puts the reduced pole near 2.
    t = uniform_nodes(0.25, 0.25)
    X, Y = weighted_impulses(A_PLANT, B_PLANT, C_PLANT, t)
    reduced = bpod(
        A_PLANT, B_PLANT, C_PLANT, X, Y, D_PLANT; n = 1, atol = 0.0, rtol = 0.0
    )
    @test maximum(real, eigvals(A_PLANT)) < 0
    @test maximum(real, eigvals(reduced.A)) > 0
end
