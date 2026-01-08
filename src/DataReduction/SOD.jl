"""
    SOD <: AbstractDRProblem

Smooth Orthogonal Decomposition (SOD) for extracting linear normal modes and natural
frequencies from time-series data.

SOD extends Proper Orthogonal Decomposition (POD) by considering temporal smoothness
in addition to spatial variance. It uses the covariance matrices of displacement and
velocity responses to form a generalized eigenvalue problem, which yields modes and
frequencies without requiring knowledge of the system's mass matrix.

# Fields
- `snapshots`: Displacement snapshot data (matrix or vector of vectors), size (n_dof, n_samples)
- `velocities`: Velocity snapshot data (same format as snapshots), or `:auto` to compute from snapshots
- `dt`: Time step for computing velocities when `velocities = :auto`
- `min_renergy`: Minimum relative energy threshold (0.0 to 1.0)
- `min_nmodes`: Minimum number of modes to retain
- `max_nmodes`: Maximum number of modes to retain
- `nmodes`: Number of modes (computed after reduction)
- `rbasis`: Reduced basis matrix (smooth orthogonal modes)
- `renergy`: Relative energy captured by the modes
- `spectrum`: Smooth orthogonal values (SOVs), which approximate ω² (squared frequencies)
- `frequencies`: Estimated natural frequencies (√spectrum)

# Constructor
    SOD(snapshots; velocities=:auto, dt=1.0, min_renergy=1.0, min_nmodes=1, max_nmodes=size(snapshots,1))
    SOD(snapshots, nmodes; velocities=:auto, dt=1.0)

# References
- Chelidze D, Zhou W. Smooth orthogonal decomposition-based vibration mode identification.
  Journal of Sound and Vibration. 2006; 292(3-5): 461-473.
  https://www.sciencedirect.com/science/article/abs/pii/S0022460X05005948
"""
mutable struct SOD <: AbstractDRProblem
    # specified
    snapshots::Any
    velocities::Any
    dt::Float64
    min_renergy::Any
    min_nmodes::Int
    max_nmodes::Int
    # computed
    nmodes::Int
    rbasis::Any
    renergy::Any
    spectrum::Any
    frequencies::Any
    # constructors
    function SOD(
            snaps;
            velocities = :auto,
            dt::Real = 1.0,
            min_renergy = 1.0,
            min_nmodes::Int = 1,
            max_nmodes::Int = _get_ndof(snaps)
        )
        nmodes = min_nmodes
        errorhandle_sod(snaps, velocities, dt, nmodes, min_renergy, min_nmodes, max_nmodes)
        return new(
            snaps, velocities, Float64(dt), min_renergy, min_nmodes, max_nmodes,
            nmodes, missing, 1.0, missing, missing
        )
    end
    function SOD(snaps, nmodes::Int; velocities = :auto, dt::Real = 1.0)
        errorhandle_sod(snaps, velocities, dt, nmodes, 0.0, nmodes, nmodes)
        return new(
            snaps, velocities, Float64(dt), 0.0, nmodes, nmodes,
            nmodes, missing, 1.0, missing, missing
        )
    end
end

# Helper to get number of degrees of freedom from snapshots
function _get_ndof(snaps::AbstractMatrix)
    return size(snaps, 1)
end

function _get_ndof(snaps::AbstractVector{<:AbstractVector})
    return length(snaps[1])
end

# Error handling for SOD
function errorhandle_sod(
        data::AbstractMatrix, velocities, dt, nmodes::Int, max_energy,
        min_nmodes, max_nmodes
    )
    @assert size(data, 1) > 1 "State vector is expected to be vector valued."
    s = minimum(size(data))
    @assert 0 < nmodes <= s "Number of modes should be in {1,2,...,$s}."
    @assert min_nmodes <= max_nmodes "Minimum number of modes must lie below maximum number of modes"
    @assert 0.0 <= max_energy <= 1.0 "Maximum relative energy must be in [0,1]"
    @assert dt > 0 "Time step dt must be positive"
    if velocities !== :auto
        @assert size(velocities) == size(data) "Velocities must have same size as snapshots"
    end
    return nothing
end

function errorhandle_sod(
        data::AbstractVector{T}, velocities, dt, nmodes::Int, max_energy, min_nmodes,
        max_nmodes
    ) where {T <: AbstractVector}
    @assert size(data[1], 1) > 1 "State vector is expected to be vector valued."
    s = min(size(data, 1), size(data[1], 1))
    @assert 0 < nmodes <= s "Number of modes should be in {1,2,...,$s}."
    @assert min_nmodes <= max_nmodes "Minimum number of modes must lie below maximum number of modes"
    @assert 0.0 <= max_energy <= 1.0 "Maximum relative energy must be in [0,1]"
    @assert dt > 0 "Time step dt must be positive"
    if velocities !== :auto
        @assert length(velocities) == length(data) "Velocities must have same length as snapshots"
    end
    return nothing
end

# Compute velocity from displacement using finite differences
function _compute_velocity(X::AbstractMatrix, dt::Real)
    # Central difference for interior points, forward/backward for boundaries
    n_dof, n_samples = size(X)
    V = similar(X)

    if n_samples == 1
        V .= 0
        return V
    elseif n_samples == 2
        # Forward difference only
        V[:, 1] = (X[:, 2] - X[:, 1]) / dt
        V[:, 2] = V[:, 1]
        return V
    end

    # Forward difference for first point
    V[:, 1] = (X[:, 2] - X[:, 1]) / dt

    # Central difference for interior points
    for i in 2:(n_samples - 1)
        V[:, i] = (X[:, i + 1] - X[:, i - 1]) / (2 * dt)
    end

    # Backward difference for last point
    V[:, n_samples] = (X[:, n_samples] - X[:, n_samples - 1]) / dt

    return V
end

function _compute_velocity(X::AbstractVector{<:AbstractVector}, dt::Real)
    X_mat = matricize(X)
    V_mat = _compute_velocity(X_mat, dt)
    return V_mat
end

# Convert velocities to matrix form
function _get_velocity_matrix(sod::SOD)
    if sod.velocities === :auto
        X = sod.snapshots isa AbstractMatrix ? sod.snapshots : matricize(sod.snapshots)
        return _compute_velocity(X, sod.dt)
    else
        return sod.velocities isa AbstractMatrix ? sod.velocities : matricize(sod.velocities)
    end
end

# Get displacement matrix
function _get_displacement_matrix(sod::SOD)
    return sod.snapshots isa AbstractMatrix ? sod.snapshots : matricize(sod.snapshots)
end

"""
    reduce!(sod::SOD)

Perform Smooth Orthogonal Decomposition on the snapshot data.

This solves the generalized eigenvalue problem:
    Σ_xx Ψ = λ Σ_vv Ψ

where Σ_xx = X X^T / N (displacement covariance) and Σ_vv = V V^T / N (velocity covariance).

The eigenvalues λ (smooth orthogonal values, SOVs) approximate ω² (squared natural frequencies).
The eigenvectors Ψ are the smooth orthogonal modes (SOMs).
"""
function reduce!(sod::SOD)
    X = _get_displacement_matrix(sod)
    V = _get_velocity_matrix(sod)

    n_dof, n_samples = size(X)

    # Compute covariance matrices
    # Σ_xx = X * X' / n_samples (displacement covariance)
    # Σ_vv = V * V' / n_samples (velocity covariance)
    Σ_xx = X * X' / n_samples
    Σ_vv = V * V' / n_samples

    # Add small regularization to Σ_vv if it's singular
    # This handles cases where velocity variance is very small
    eps_reg = 1.0e-10 * maximum(abs, Σ_vv)
    if eps_reg < 1.0e-14
        eps_reg = 1.0e-14
    end
    Σ_vv_reg = Σ_vv + eps_reg * I

    # Solve generalized eigenvalue problem: Σ_xx Ψ = λ Σ_vv Ψ
    # This is equivalent to finding eigenvalues of Σ_vv^(-1) Σ_xx
    # For numerical stability, we use eigen on the symmetric pencil
    eig_result = eigen(Symmetric(Σ_xx), Symmetric(Σ_vv_reg))

    # Eigenvalues are the SOVs (smooth orthogonal values)
    # They should be sorted in decreasing order (largest eigenvalue = highest energy)
    sovs = real.(eig_result.values)
    soms = real.(eig_result.vectors)

    # Sort by eigenvalue magnitude (largest first)
    perm = sortperm(sovs, rev = true)
    sovs = sovs[perm]
    soms = soms[:, perm]

    # Determine truncation based on energy criterion
    sod.nmodes, sod.renergy = determine_truncation(
        sovs, sod.min_nmodes, sod.min_renergy, sod.max_nmodes
    )

    # Store results
    sod.rbasis = soms[:, 1:(sod.nmodes)]
    sod.spectrum = sovs

    # Compute natural frequencies from SOVs
    # SOVs approximate ω², so frequencies are √SOVs
    # Only take sqrt of positive values
    sod.frequencies = sqrt.(max.(sovs[1:(sod.nmodes)], 0.0))

    return nothing
end

function Base.show(io::IO, sod::SOD)
    print(io, "SOD \n")
    print(io, "Reduction Order = ", sod.nmodes, "\n")
    n_dof = _get_ndof(sod.snapshots)
    n_samples = sod.snapshots isa AbstractMatrix ? size(sod.snapshots, 2) :
        length(sod.snapshots)
    print(io, "Snapshot size = (", n_dof, ",", n_samples, ")\n")
    print(io, "Relative Energy = ", sod.renergy, "\n")
    if !ismissing(sod.frequencies)
        print(io, "Estimated frequencies = ", sod.frequencies, "\n")
    end
    return nothing
end
