using PrecompileTools

@setup_workload begin
    using LinearAlgebra: qr

    @compile_workload begin
        # Precompile POD with matrix input (most common use case)
        mat_snapshots = randn(20, 10)
        pod_mat = POD(mat_snapshots, 3)
        reduce!(pod_mat, SVD())

        # Precompile POD with Vector{Vector} input
        vec_snapshots = [randn(20) for _ in 1:10]
        pod_vec = POD(vec_snapshots, 3)
        reduce!(pod_vec, TSVD())

        # Precompile RSVD algorithm
        pod_rsvd = POD(mat_snapshots, 3)
        reduce!(pod_rsvd, RSVD(2))

        # Precompile deim_interpolation_indices
        basis = Matrix(qr(randn(20, 5)).Q)
        deim_interpolation_indices(basis)
    end
end
