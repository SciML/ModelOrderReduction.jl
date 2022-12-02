using Test, ModelOrderReduction
using DifferentialEquations
const MOR = ModelOrderReduction
function lorenz_prob()
    function lorenz!(du, u, p, t)
        du[1] = p[1] * (u[2] - u[1])
        du[2] = u[1] * (p[2] - u[3]) - u[2]
        du[3] = u[1] * u[2] - p[3] * u[3]
    end

    u0 = [1, 0, 0]
    p = [10, 28, 8 / 3]
    tspan = (0, 100)
    prob = ODEProblem(lorenz!, u0, tspan, p)
    sol = solve(prob, Tsit5())
    sol
end

@testset "POD-Utils" begin
    solution = lorenz_prob()
    VoV = solution.u
    M = ModelOrderReduction.matricize(VoV)
    @test size(M, 1) == length(VoV[1]) # Parameters
    @test size(M, 2) == length(VoV) # Time
end

@testset "POD - Attractor Test" begin
    sol = lorenz_prob()
    solution = Array(sol)

    order = 2
    solver = MOR.SVD()
    matrix_reducer = POD(solution, order)
    snapshot_reducer = POD(sol.u, order)
    reduce!(matrix_reducer, solver)
    reduce!(snapshot_reducer, solver)

    @test all(matrix_reducer.rbasis .≈ snapshot_reducer.rbasis)
    @test matrix_reducer.renergy ≈ snapshot_reducer.renergy

    @test size(matrix_reducer.rbasis, 2) == matrix_reducer.nmodes
    @test size(matrix_reducer.rbasis, 1) == size(solution, 1)
    @test matrix_reducer.renergy > 0.9

    reducer = POD(solution, min_nmodes = 1, max_nmodes = 2, min_renergy = 0.1)
    reduce!(reducer, solver)
    @test reducer.renergy > 0.1
    @test reducer.nmodes == 1

    order = 2
    solver = MOR.TSVD()
    reducer = POD(solution, order)
    reduce!(reducer, solver)

    @test size(reducer.rbasis, 2) == reducer.nmodes
    @test size(reducer.rbasis, 1) == size(solution, 1)
    @test reducer.renergy > 0.7

    order = 2
    solver = MOR.RSVD()
    reducer = POD(solution, order)
    reduce!(reducer, solver)

    @test size(reducer.rbasis, 2) == reducer.nmodes
    @test size(reducer.rbasis, 1) == size(solution, 1)
    @test reducer.renergy > 0.7
end
