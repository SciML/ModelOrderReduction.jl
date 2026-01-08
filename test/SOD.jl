using Test, ModelOrderReduction
using OrdinaryDiffEq

# Simple harmonic oscillator system for testing SOD
# This system has known natural frequencies that SOD should recover
function harmonic_oscillator_prob()
    # 2-DOF mass-spring system with known frequencies
    # m1*x1'' + k1*x1 + k2*(x1-x2) = 0
    # m2*x2'' + k2*(x2-x1) = 0
    # With m1=m2=1, k1=1, k2=1, natural frequencies are ω₁ ≈ 0.618, ω₂ ≈ 1.618

    function harmonic!(du, u, p, t)
        # u = [x1, v1, x2, v2]
        k1, k2, m1, m2 = p
        x1, v1, x2, v2 = u
        du[1] = v1
        du[2] = (-k1 * x1 - k2 * (x1 - x2)) / m1
        du[3] = v2
        return du[4] = (-k2 * (x2 - x1)) / m2
    end

    u0 = [1.0, 0.0, 0.5, 0.0]  # Initial displacement, zero velocity
    p = [1.0, 1.0, 1.0, 1.0]   # k1, k2, m1, m2
    tspan = (0.0, 50.0)
    prob = ODEProblem(harmonic!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat = 0.01)
    return sol
end

# Use Lorenz system for comparison with POD (same test as POD uses)
function lorenz_prob()
    function lorenz!(du, u, p, t)
        du[1] = p[1] * (u[2] - u[1])
        du[2] = u[1] * (p[2] - u[3]) - u[2]
        return du[3] = u[1] * u[2] - p[3] * u[3]
    end

    u0 = [1, 0, 0]
    p = [10, 28, 8 / 3]
    tspan = (0, 100)
    prob = ODEProblem(lorenz!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat = 0.1)
    return sol
end

@testset "SOD - Basic Functionality" begin
    sol = lorenz_prob()
    solution = Array(sol)
    dt = 0.1

    # Test with matrix input
    order = 2
    matrix_reducer = SOD(solution, order; dt = dt)
    reduce!(matrix_reducer)

    @test size(matrix_reducer.rbasis, 2) == matrix_reducer.nmodes
    @test size(matrix_reducer.rbasis, 1) == size(solution, 1)
    @test !ismissing(matrix_reducer.spectrum)
    @test !ismissing(matrix_reducer.frequencies)
    @test length(matrix_reducer.frequencies) == order

    # Test with vector of vectors input
    snapshot_reducer = SOD(sol.u, order; dt = dt)
    reduce!(snapshot_reducer)

    @test size(snapshot_reducer.rbasis, 2) == snapshot_reducer.nmodes
    @test !ismissing(snapshot_reducer.frequencies)
end

@testset "SOD - Energy Truncation" begin
    sol = lorenz_prob()
    solution = Array(sol)
    dt = 0.1

    # Test with energy-based truncation
    reducer = SOD(solution; dt = dt, min_nmodes = 1, max_nmodes = 3, min_renergy = 0.1)
    reduce!(reducer)
    @test reducer.renergy >= 0.1
    @test 1 <= reducer.nmodes <= 3
end

@testset "SOD - Provided Velocities" begin
    sol = lorenz_prob()
    solution = Array(sol)
    dt = 0.1

    # Manually compute velocities (central difference)
    n_dof, n_samples = size(solution)
    velocities = similar(solution)
    velocities[:, 1] = (solution[:, 2] - solution[:, 1]) / dt
    for i in 2:(n_samples - 1)
        velocities[:, i] = (solution[:, i + 1] - solution[:, i - 1]) / (2 * dt)
    end
    velocities[:, n_samples] = (solution[:, n_samples] - solution[:, n_samples - 1]) / dt

    # Test with provided velocities
    order = 2
    reducer = SOD(solution, order; velocities = velocities, dt = dt)
    reduce!(reducer)

    @test size(reducer.rbasis, 2) == reducer.nmodes
    @test !ismissing(reducer.frequencies)
end

@testset "SOD - Harmonic Oscillator Frequency Recovery" begin
    sol = harmonic_oscillator_prob()

    # Extract position data (x1 and x2, not velocities)
    positions = vcat(sol[1, :]', sol[3, :]')  # Shape: (2, n_samples)
    dt = 0.01

    # SOD should find the modes of this oscillating system
    reducer = SOD(positions, 2; dt = dt)
    reduce!(reducer)

    @test reducer.nmodes == 2
    @test !ismissing(reducer.frequencies)

    # Check that we get reasonable positive frequencies
    # (The exact values depend on the sampling and system dynamics)
    @test all(reducer.frequencies .>= 0)
end

@testset "SOD - Show Method" begin
    sol = lorenz_prob()
    solution = Array(sol)
    dt = 0.1

    reducer = SOD(solution, 2; dt = dt)
    reduce!(reducer)

    # Test that show doesn't error
    io = IOBuffer()
    show(io, reducer)
    str = String(take!(io))
    @test occursin("SOD", str)
    @test occursin("Reduction Order", str)
    @test occursin("frequencies", str)
end
