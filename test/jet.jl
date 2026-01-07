using Test, JET
using ModelOrderReduction
using LinearAlgebra: qr

@testset "JET Static Analysis" begin
    # Create test data
    n = 20  # state dimension
    m = 10  # number of snapshots
    snapshot_matrix = Float64[sin(i * j / n) for i in 1:n, j in 1:m]
    snapshot_vov = [Float64[sin(i * j / n) for i in 1:n] for j in 1:m]

    # Create an orthonormal basis for deim_interpolation_indices
    Q, _ = qr(snapshot_matrix)
    deim_basis = Matrix(Q[:, 1:5])

    @testset "deim_interpolation_indices type stability" begin
        rep = JET.report_call(ModelOrderReduction.deim_interpolation_indices, (Matrix{Float64},))
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "matricize type stability" begin
        rep = JET.report_call(ModelOrderReduction.matricize, (Vector{Vector{Float64}},))
        @test length(JET.get_reports(rep)) == 0
    end

    @testset "POD constructor type stability" begin
        # Matrix constructor
        rep1 = JET.report_call(ModelOrderReduction.POD, (Matrix{Float64}, Int))
        @test length(JET.get_reports(rep1)) == 0

        # Vector{Vector} constructor
        rep2 = JET.report_call(ModelOrderReduction.POD, (Vector{Vector{Float64}}, Int))
        @test length(JET.get_reports(rep2)) == 0
    end

    @testset "reduce! with SVD type stability" begin
        pod = POD(snapshot_matrix, 3)
        rep = JET.report_call(ModelOrderReduction.reduce!, (typeof(pod), typeof(SVD())))
        @test length(JET.get_reports(rep)) == 0
    end
end
