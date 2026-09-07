"""
    $(TYPEDEF)

Reduced continuous-time LTI model produced by [`baltrunc`](@ref).

The reduced dynamics are
```math
\\dot x_r = A_r x_r + B_r u,\\qquad y = C_r x_r + D_r u,
```
with reconstruction ``x \\approx T_r x_r`` and reduction ``x_r = S x`` for plain
truncation. When residualization is used, ``T_r`` is still the balanced truncation
map for the retained states; the static contribution of the discarded states is folded
into ``(A_r, B_r, C_r, D_r)`` rather than into ``T_r``.

# Fields
$(TYPEDFIELDS)
"""
struct BalancedTruncation{T <: AbstractFloat}
    "reduced state matrix ``A_r``"
    A::Matrix{T}
    "reduced input matrix ``B_r``"
    B::Matrix{T}
    "reduced output matrix ``C_r``"
    C::Matrix{T}
    "feedthrough matrix ``D_r``"
    D::Matrix{T}
    "retained Hankel singular values"
    hsv::Vector{T}
    "right projector ``T_r`` with ``x \\approx T_r x_r`` under plain truncation"
    Tr::Matrix{T}
    "left projector ``S`` with ``x_r = S x``"
    S::Matrix{T}
end

function Base.show(io::IO, bt::BalancedTruncation)
    r = size(bt.A, 1)
    nu = size(bt.B, 2)
    ny = size(bt.C, 1)
    return print(io, "BalancedTruncation(order = $r, inputs = $nu, outputs = $ny)")
end

"""
    $(TYPEDSIGNATURES)

Return a square-root factor ``L`` of the positive-semidefinite matrix ``W`` such that
``W \\approx L L'``. Tiny negative eigenvalues from roundoff are clipped to zero.
"""
function _psd_sqrt(W::AbstractMatrix{T}; atol::Real = 0) where {T <: AbstractFloat}
    F = eigen(Symmetric(Matrix{T}(W)))
    λ = F.values
    λmax = maximum(abs, λ)
    thresh = max(T(atol), eps(T) * λmax)
    kept = findall(≥(thresh), λ)
    isempty(kept) && return zeros(T, size(W, 1), 0)
    return F.vectors[:, kept] * Diagonal(sqrt.(λ[kept]))
end

"""
    $(TYPEDSIGNATURES)

Compute continuous-time controllability and observability Gramians of ``(A,B,C)``.

Solves
```math
A P_c + P_c A' + B B' = 0,\\qquad A' P_o + P_o A + C' C = 0
```
using `LinearAlgebra.lyap`. The open-loop matrix `A` must be Hurwitz.
"""
function _gramians(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
    Pc = lyap(A, B * B')
    Po = lyap(A', C' * C)
    return Symmetric(Pc), Symmetric(Po)
end

function _select_order(σ::AbstractVector{T}, n, atol::Real, rtol::Real) where {T}
    isempty(σ) && return 0
    if n !== nothing
        n isa Integer || throw(ArgumentError("n must be an Integer or `nothing`"))
        n < 0 && throw(ArgumentError("n must be nonnegative"))
        return min(Int(n), length(σ))
    end
    σmax = σ[1]
    return count(s -> s >= T(atol) && s >= T(rtol) * σmax, σ)
end

function _assert_hurwitz(A::AbstractMatrix{T}) where {T <: AbstractFloat}
    if maximum(real, eigvals(A)) >= zero(T)
        throw(ArgumentError("A must be Hurwitz (all eigenvalues with negative real part) for continuous-time balanced truncation"))
    end
    return nothing
end

"""
    baltrunc(A, B, C, D = nothing; n = nothing, atol = 0, rtol = 1e-3,
             residual = false) -> BalancedTruncation

Balanced truncation of a continuous-time linear time-invariant system
```math
\\dot x = A x + B u,\\qquad y = C x + D u.
```

The square-root method balances the controllability and observability Gramians and
truncates states with small Hankel singular values. If `n` is omitted, the reduced
order keeps singular values at least `atol` and at least `rtol` times the largest
singular value.

When `residual = true`, truncated states are eliminated by residualization
(singular perturbation) so that the static gain is matched.

Only real floating-point systems are supported. `A` must be Hurwitz.

# Arguments
- `A::AbstractMatrix`: state matrix.
- `B::AbstractVecOrMat`: input matrix (vectors are treated as single-input).
- `C::AbstractMatrix`: output matrix.
- `D`: feedthrough matrix. Defaults to a zero matrix conforming to `(C, B)`.

# Keywords
- `n = nothing`: reduced order. When `nothing`, choose the order from `atol` / `rtol`.
- `atol = 0`: absolute Hankel-singular-value cutoff.
- `rtol = 1e-3`: relative Hankel-singular-value cutoff.
- `residual = false`: use residualization instead of plain truncation.

# Returns
- [`BalancedTruncation`](@ref): reduced matrices, retained Hankel singular values, and
  projectors.

# Throws
- `ArgumentError`: if the system dimensions are inconsistent, `A` is not square, `A` is
  not Hurwitz, no positive Hankel singular values remain, or `n` is invalid.

# Examples
```jldoctest
julia> using ModelOrderReduction

julia> A = [-1.0 0.0; 0.0 -2.0]; B = reshape([1.0, 1.0], 2, 1); C = [1.0 1.0];

julia> bt = baltrunc(A, B, C; n = 1);

julia> size(bt.A)
(1, 1)
```

# References
- Moore, B. C. (1981). Principal component analysis in linear systems. *IEEE
  Transactions on Automatic Control*.
- Antoulas, A. C. (2005). *Approximation of Large-Scale Dynamical Systems*. SIAM.
"""
function baltrunc(
        A::AbstractMatrix,
        B::AbstractVecOrMat,
        C::AbstractMatrix,
        D::Union{AbstractMatrix, Nothing} = nothing;
        n = nothing,
        atol::Real = 0,
        rtol::Real = 1.0e-3,
        residual::Bool = false
)
    TA = float(promote_type(eltype(A), eltype(B), eltype(C)))
    TA <: AbstractFloat ||
        throw(ArgumentError("baltrunc currently supports real floating-point systems only"))
    A = Matrix{TA}(A)
    B = Matrix{TA}(B isa AbstractVector ? reshape(B, length(B), 1) : B)
    C = Matrix{TA}(C)
    nx = size(A, 1)
    size(A, 2) == nx || throw(ArgumentError("A must be square"))
    size(B, 1) == nx || throw(ArgumentError("B must have the same number of rows as A"))
    size(C, 2) == nx || throw(ArgumentError("C must have the same number of columns as A"))
    if D === nothing
        D = zeros(TA, size(C, 1), size(B, 2))
    else
        D = Matrix{TA}(D)
        size(D) == (size(C, 1), size(B, 2)) ||
            throw(ArgumentError("D must have size (size(C, 1), size(B, 2))"))
    end
    if n !== nothing
        n isa Integer || throw(ArgumentError("n must be an Integer or `nothing`"))
        n < 1 && throw(ArgumentError("n must be a positive Integer"))
    end

    _assert_hurwitz(A)
    Pc, Po = _gramians(A, B, C)
    Lc = _psd_sqrt(Pc)
    Lo = _psd_sqrt(Po)
    (size(Lc, 2) == 0 || size(Lo, 2) == 0) &&
        throw(ArgumentError("controllability or observability Gramian has no positive eigenvalues; check stability and (A,B,C)"))

    U, σ, V = svd(Lo' * Lc; full = false)
    # Drop numerically zero HSVs before forming balancing transforms.
    σtol = max(eps(TA)^(2 / 3) * (isempty(σ) ? zero(TA) : σ[1]), TA(atol))
    keep = findall(>(σtol), σ)
    isempty(keep) &&
        throw(ArgumentError("no Hankel singular values satisfy the truncation tolerances"))
    U = U[:, keep]
    σ = σ[keep]
    V = V[:, keep]

    r = _select_order(σ, n, atol, rtol)
    r == 0 &&
        throw(ArgumentError("no Hankel singular values satisfy the truncation tolerances"))
    if n !== nothing && Int(n) > length(σ)
        throw(ArgumentError("requested order n = $n exceeds the $(length(σ)) positive Hankel singular values"))
    end

    i1 = 1:r
    σr = σ[i1]
    Σinvsqrt = Diagonal(inv.(sqrt.(σr)))
    Tr = Lc * V[:, i1] * Σinvsqrt          # x ≈ Tr xr
    Sl = Σinvsqrt * U[:, i1]' * Lo'        # xr = Sl x

    if residual && r < length(σ)
        # Balanced realization on the retained positive-HSV subspace, then residualize.
        Σinvsqrt_full = Diagonal(inv.(sqrt.(σ)))
        Tfull = Lc * V * Σinvsqrt_full
        Sfull = Σinvsqrt_full * U' * Lo'
        Abal = Sfull * A * Tfull
        Bbal = Sfull * B
        Cbal = C * Tfull
        i2 = (r + 1):length(σ)
        A11 = Abal[i1, i1]
        A12 = Abal[i1, i2]
        A21 = Abal[i2, i1]
        A22 = Abal[i2, i2]
        B1 = Bbal[i1, :]
        B2 = Bbal[i2, :]
        C1 = Cbal[:, i1]
        C2 = Cbal[:, i2]
        A22fac = factorize(-A22)
        A2221 = A22fac \ A21
        Ar = A11 + A12 * A2221
        Br = B1 + A12 * (A22fac \ B2)
        Cr = C1 + C2 * A2221
        Dr = D + C2 * (A22fac \ B2)
        return BalancedTruncation{TA}(Ar, Br, Cr, Dr, Vector{TA}(σr), Tr, Sl)
    else
        Ar = Sl * A * Tr
        Br = Sl * B
        Cr = C * Tr
        return BalancedTruncation{TA}(Ar, Br, Cr, D, Vector{TA}(σr), Tr, Sl)
    end
end
