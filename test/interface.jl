using Test, ModelOrderReduction

@testset "Public POD interface" begin
    matrix_snapshots = [3.0 0.0; 0.0 1.0]
    vector_snapshots = [[3.0, 0.0], [0.0, 1.0]]

    for snapshots in (matrix_snapshots, vector_snapshots)
        for alg in (SVD(), TSVD(), RSVD())
            pod = POD(snapshots, 1)
            @test reduce!(pod, alg) === nothing
            @test size(pod.rbasis) == (2, 1)
            @test length(pod.spectrum) >= pod.nmodes
            @test pod.nmodes == 1
            @test 0.0 < pod.renergy <= 1.0
        end
    end
end
