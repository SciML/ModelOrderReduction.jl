using ModelOrderReduction, Test

@testset "POD $T" for T in (Float32, Float64)
    n_eq = 10 # number of equations
    n_snapshot = 6 # number of snapshots
    snapshots = rand(T, n_eq, n_snapshot)

    dim₁ = 3 # POD dimension
    pod_basis₁, singular_vals₁ = pod(snapshots, dim₁)
    @test size(pod_basis₁) == (n_eq, dim₁)
    @test size(singular_vals₁, 1) == dim₁
    @test eltype(pod_basis₁) == T
    @test eltype(singular_vals₁) == T

    dim₂ = 8 # larger than the number of snapshots
    pod_basis₂, singular_vals₂ = @test_logs (:warn,) pod(snapshots, dim₂)
    @test size(pod_basis₂) == (n_eq, n_snapshot)
    @test size(singular_vals₂, 1) == n_snapshot
    @test eltype(pod_basis₂) == T
    @test eltype(singular_vals₂) == T
end
