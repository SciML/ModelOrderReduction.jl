using PrecompileTools: PrecompileTools, @compile_workload, @setup_workload

@setup_workload begin
    # Setup code - create minimal test data
    n = 20  # small state dimension
    m = 10  # small number of snapshots
    snapshot_matrix = Float64[sin(i * j / n) for i in 1:n, j in 1:m]
    snapshot_vov = [Float64[sin(i * j / n) for i in 1:n] for j in 1:m]

    # Create an orthonormal basis for deim_interpolation_indices
    Q, _ = qr(snapshot_matrix)
    deim_basis = Matrix(Q[:, 1:5])

    @compile_workload begin
        # POD construction with matrix input
        pod_mat = POD(snapshot_matrix, 3)

        # POD construction with vector of vectors
        pod_vov = POD(snapshot_vov, 3)

        # reduce! with SVD algorithm
        reduce!(pod_mat, SVD())

        # reduce! with TSVD algorithm
        pod_tsvd = POD(snapshot_matrix, 3)
        reduce!(pod_tsvd, TSVD())

        # reduce! with RSVD algorithm
        pod_rsvd = POD(snapshot_matrix, 3)
        reduce!(pod_rsvd, RSVD())

        # deim_interpolation_indices (core DEIM algorithm)
        deim_interpolation_indices(deim_basis)
    end
end
