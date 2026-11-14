"""
    $(TYPEDSIGNATURES)

Balanced proper orthogonal decomposition of a continuous-time linear system
```math
\\dot x = A x + B u,\\qquad y = C x + D u,
```
from already weighted primal and adjoint snapshot factors.

Let `A` be `nx × nx`, `B` be `nx × nu`, `C` be `ny × nx`, and `D` be `ny × nu`.
A vector `B` has length `nx` and is treated as `nu = 1`. The snapshot factors `X`
and `Y` are `nx × nX` and `nx × nY`. The two column counts need not agree.

The factors follow Rowley (2005). Column `k` of `X` is a primal impulse response
`exp(A t) b_j` already multiplied by the square root of its quadrature weight, and
column `k` of `Y` is the corresponding adjoint impulse `exp(A' t) c_i` with the
same factor. The empirical Gramians are then ``X X'`` and ``Y Y'``. `bpod` does
not form those products, and it does not multiply the columns by the weights. A
second application of the weights squares them.

The singular value decomposition ``Y' X = U \\Sigma V'`` supplies the projectors.
After the numerical-rank cutoff below, a leading part ``\\Sigma_r`` is retained:
```math
T_r = X V_r \\Sigma_r^{-1/2},\\qquad
S = \\Sigma_r^{-1/2} U_r' Y',\\qquad
A_r = S A T_r,\\; B_r = S B,\\; C_r = C T_r,\\; D_r = D.
```
`D` is copied, or replaced by a zero matrix of size `ny × nu` when it is omitted.
The state is reconstructed by ``x \\approx T_r x_r`` and reduced by ``x_r = S x``.
A distinct singular value fixes its singular vectors up to one shared sign.
Repeated singular values fix only their subspace, which may be rotated. Changing
the sign of column ``j`` of ``T_r`` and row ``j`` of ``S`` together leaves
``S T_r``, the Hankel values, and the input-output map unchanged.

Forming `Y' * X` produces a dense `nY × nX` matrix, and decomposing it dominates
the cost after the factors are available. That matrix has `nY * nX` entries
whether or not `A`, `B`, and `C` use the sparse or banded storage below.

Singular values no larger than `max(eps(T)^(2/3) * σ₁, atol)` are dropped before
`1 / sqrt(σ)` is formed. If `n` is `nothing`, the reduced order is the number of
remaining values that are at least `rtol` times the largest. An explicit positive
integer `n` ignores `rtol` and must not exceed the number of values that survived
the cutoff. `hsv` contains only the retained values, in descending order. Pass
`n = nothing`, `atol = 0`, and `rtol = 0` to keep every value above the cutoff.

`A`, `B`, `C`, `D`, `X`, and `Y` are promoted to one floating-point type before
any arithmetic. `Float32` and `Float64` are the supported, tested element types.
Standard integer element types become `Float64` through `float` of that promoted
type. Mixed `Float32` and `Float64` inputs use `Float64`. Complex inputs are
rejected. A sparse matrix, the adjoint or transpose of one, and a `Diagonal`,
`Bidiagonal`, `Tridiagonal`, or `SymTridiagonal` matrix keep that storage. A
`Symmetric` or `Hermitian` wrapper, including one around a sparse matrix, has no
such conversion and is not a supported sparse operator. The reduced products are
`S * (A * Tr)`, `S * B`, and `C * Tr`.

This is plain truncation. It does not residualize, generate time nodes, choose
quadrature weights, or integrate the impulses. `A` is not required to be Hurwitz,
and a Hurwitz `A` does not make the reduced state matrix Hurwitz. Finite-horizon
factors can produce an unstable reduced model. When a stable reduced model is
required, inspect the eigenvalues of its state matrix. If `A` is not Hurwitz, the
singular values from a finite horizon are not the infinite-horizon Hankel singular
values of a stable system. The returned Hankel values are the retained singular
values of `Y' * X`. They approximate the infinite-horizon Hankel singular values
only to the extent that `X` and `Y` approximate factors of those Gramians. A finite
response horizon omits the tail, and the quadrature rule adds its own error. Both
errors pass into `hsv` and the reduced matrices. For a large system, compute the
impulse responses with a matrix-free exponential or an ODE solver and pass the
weighted factors in. A dense matrix exponential is appropriate only for a small
dense `A`.

# Arguments
- `A::AbstractMatrix`: state matrix, size `nx × nx`.
- `B::AbstractVecOrMat`: input matrix, size `nx × nu`. A vector is one input.
- `C::AbstractMatrix`: output matrix, size `ny × nx`.
- `X::AbstractMatrix`: weighted primal snapshots, size `nx × nX`.
- `Y::AbstractMatrix`: weighted adjoint snapshots, size `nx × nY`.
- `D`: feedthrough matrix, size `ny × nu`. Defaults to zero.

# Keywords
- `n = nothing`: reduced order. When `nothing`, choose the order from `atol` / `rtol`.
- `atol = 0`: absolute Hankel-singular-value cutoff.
- `rtol = 1e-3`: relative Hankel-singular-value cutoff, used only when `n` is `nothing`.

# Returns
- [`BalancedTruncation`](@ref): reduced matrices `A`, `B`, `C`, and `D`; the retained
  approximate Hankel singular values `hsv`; and the projectors `Tr` and `S`.

# Throws
- `ArgumentError`: if the inputs do not promote to a real floating-point type,
  contain a non-finite value, or have inconsistent dimensions; if `X` or `Y` has
  no columns; if `n` is invalid or larger than the number of singular values above
  the cutoff; or if no positive Hankel singular values satisfy the tolerances.

# Examples
```jldoctest
julia> using ModelOrderReduction

julia> A = [-1.0 0.0; 0.0 -2.0]; B = [1.0, 0.5]; C = [1.0 1.0];

julia> X = [1.0 0.0; 0.0 1.0]; Y = [1.0 0.0; 0.0 1.0]; # already weighted

julia> rom = bpod(A, B, C, X, Y; n = 1);

julia> size(rom.A)
(1, 1)
```

# References
- Rowley, C. W. (2005). Model reduction for fluids, using balanced proper orthogonal
  decomposition. *International Journal of Bifurcation and Chaos*.
"""
function bpod(
        A::AbstractMatrix,
        B::AbstractVecOrMat,
        C::AbstractMatrix,
        X::AbstractMatrix,
        Y::AbstractMatrix,
        D::Union{AbstractMatrix, Nothing} = nothing;
        n = nothing,
        atol::Real = 0,
        rtol::Real = 1.0e-3
    )
    TD = D === nothing ? eltype(A) : eltype(D)
    T = float(promote_type(eltype(A), eltype(B), eltype(C), eltype(X), eltype(Y), TD))
    T <: AbstractFloat ||
        throw(ArgumentError("bpod currently supports real floating-point systems only"))
    B = B isa AbstractVector ? reshape(B, length(B), 1) : B
    nx = size(A, 1)
    size(A, 2) == nx || throw(ArgumentError("A must be square"))
    size(B, 1) == nx || throw(ArgumentError("B must have the same number of rows as A"))
    size(C, 2) == nx || throw(ArgumentError("C must have the same number of columns as A"))
    size(X, 1) == nx || throw(ArgumentError("X must have the same number of rows as A"))
    size(Y, 1) == nx || throw(ArgumentError("Y must have the same number of rows as A"))
    size(X, 2) > 0 || throw(ArgumentError("X must have at least one column"))
    size(Y, 2) > 0 || throw(ArgumentError("Y must have at least one column"))
    if D !== nothing && size(D) != (size(C, 1), size(B, 2))
        throw(ArgumentError("D must have size (size(C, 1), size(B, 2))"))
    end
    if n !== nothing
        n isa Integer || throw(ArgumentError("n must be an Integer or `nothing`"))
        n < 1 && throw(ArgumentError("n must be a positive Integer"))
    end

    A = _coerce_operator(A, T)
    B = _coerce_operator(B, T)
    C = _coerce_operator(C, T)
    X = X isa Matrix{T} ? X : Matrix{T}(X)
    Y = Y isa Matrix{T} ? Y : Matrix{T}(Y)
    if D === nothing
        D = zeros(T, size(C, 1), size(B, 2))
    else
        D = D isa Matrix{T} ? D : Matrix{T}(D)
    end
    _assert_finite("A", A)
    _assert_finite("B", B)
    _assert_finite("C", C)
    _assert_finite("D", D)
    _assert_finite("X", X)
    _assert_finite("Y", Y)

    product = Matrix{T}(Y' * X)
    _assert_finite("Y' * X", product)
    F = svd(product; full = false)
    # Same numerical-rank cutoff as `baltrunc`, applied before `inv(sqrt(σ))`.
    σ = F.S
    σtol = max(eps(T)^(2 / 3) * (isempty(σ) ? zero(T) : σ[1]), T(atol))
    keep = findall(>(σtol), σ)
    isempty(keep) &&
        throw(ArgumentError("no Hankel singular values satisfy the truncation tolerances"))
    U = F.U[:, keep]
    σ = σ[keep]
    V = F.V[:, keep]

    r = _select_order(σ, n, atol, rtol)
    r == 0 &&
        throw(ArgumentError("no Hankel singular values satisfy the truncation tolerances"))
    if n !== nothing && Int(n) > length(σ)
        throw(ArgumentError("requested order n = $n exceeds the $(length(σ)) positive Hankel singular values"))
    end

    σr = σ[1:r]
    Σinvsqrt = Diagonal(inv.(sqrt.(σr)))
    Tr = Matrix{T}(X * (V[:, 1:r] * Σinvsqrt))
    Sl = Matrix{T}((Σinvsqrt * U[:, 1:r]') * Y')
    # Associate as `S * (A * Tr)` so a sparse or banded `A` is not materialized.
    Ar = Matrix{T}(Sl * (A * Tr))
    Br = Matrix{T}(Sl * B)
    Cr = Matrix{T}(C * Tr)
    return BalancedTruncation{T}(Ar, Br, Cr, Matrix{T}(D), Vector{T}(σr), Tr, Sl)
end

function _coerce_operator(A::AbstractMatrix, ::Type{T}) where {T <: AbstractFloat}
    eltype(A) === T && return A
    if A isa LinearAlgebra.Adjoint
        return adjoint(_coerce_operator(parent(A), T))
    elseif A isa LinearAlgebra.Transpose
        return transpose(_coerce_operator(parent(A), T))
    elseif A isa SparseMatrixCSC
        return SparseMatrixCSC{T, Int}(A)
    elseif A isa Diagonal
        return Diagonal(Vector{T}(A.diag))
    elseif A isa LinearAlgebra.Bidiagonal
        return LinearAlgebra.Bidiagonal(Vector{T}(A.dv), Vector{T}(A.ev), A.uplo)
    elseif A isa LinearAlgebra.Tridiagonal
        return LinearAlgebra.Tridiagonal(Vector{T}(A.dl), Vector{T}(A.d), Vector{T}(A.du))
    elseif A isa LinearAlgebra.SymTridiagonal
        return LinearAlgebra.SymTridiagonal(Vector{T}(A.dv), Vector{T}(A.ev))
    else
        return Matrix{T}(A)
    end
end

# Sparse `all` inspects stored entries and one structural zero. Banded storage and
# adjoint or transpose wrappers are reduced to those stored entries. Iterating the
# matrix would also visit its structural zeros.
function _stored_entries_finite(A::AbstractArray)
    if A isa LinearAlgebra.Adjoint || A isa LinearAlgebra.Transpose
        return _stored_entries_finite(parent(A))
    elseif A isa Diagonal
        return all(isfinite, A.diag)
    elseif A isa LinearAlgebra.Bidiagonal
        return all(isfinite, A.dv) && all(isfinite, A.ev)
    elseif A isa LinearAlgebra.Tridiagonal
        return all(isfinite, A.dl) && all(isfinite, A.d) && all(isfinite, A.du)
    elseif A isa LinearAlgebra.SymTridiagonal
        return all(isfinite, A.dv) && all(isfinite, A.ev)
    else
        return all(isfinite, A)
    end
end

function _assert_finite(name::AbstractString, A::AbstractArray)
    _stored_entries_finite(A) || throw(ArgumentError("$name contains non-finite entries"))
    return nothing
end
