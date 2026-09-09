"""
    $(TYPEDEF)

Reduced model learned by [`opinf`](@ref).

With reduced coordinates ``x_r = V^\\top x``, the inferred dynamics are
```math
\\dot x_r = c + A x_r + B u + H\\,\\mathrm{quad}(x_r),
```
where ``\\mathrm{quad}`` stacks the unique quadratic monomials
``x_i x_j`` for ``1 \\le i \\le j \\le r`` in row-major lower-triangular order
``(x_1^2, x_1 x_2, \\ldots, x_1 x_r, x_2^2, \\ldots, x_r^2)``.

# Fields
$(TYPEDFIELDS)
"""
struct OperatorInferenceModel{T <: AbstractFloat}
    "POD / trial basis ``V`` with orthonormal columns"
    basis::Matrix{T}
    "linear operator ``A``, or `nothing` if not inferred"
    A::Union{Nothing, Matrix{T}}
    "quadratic operator ``H`` with `size(H, 2) == r(r + 1) ÷ 2`, or `nothing`"
    H::Union{Nothing, Matrix{T}}
    "input operator ``B``, or `nothing` if not inferred"
    B::Union{Nothing, Matrix{T}}
    "constant term ``c``, or `nothing` if not inferred"
    c::Union{Nothing, Vector{T}}
end

function Base.show(io::IO, m::OperatorInferenceModel)
    r = size(m.basis, 2)
    terms = String[]
    m.c === nothing || push!(terms, "c")
    m.A === nothing || push!(terms, "A")
    m.B === nothing || push!(terms, "B")
    m.H === nothing || push!(terms, "H")
    return print(io, "OperatorInferenceModel(order = $r, terms = $(join(terms, ", ")))")
end

"""
    $(TYPEDSIGNATURES)

Fill `out` with the unique quadratic monomials of `x` in row-major lower-triangular
order ``(x_1^2, x_1 x_2, \\ldots, x_1 x_r, x_2^2, \\ldots, x_r^2)``.
"""
function quadratic_monomials!(out::AbstractVector{To}, x::AbstractVector{Tx}) where {To, Tx}
    r = length(x)
    length(out) == (r * (r + 1)) ÷ 2 ||
        throw(DimensionMismatch("quadratic monomial buffer has wrong length"))
    k = 1
    @inbounds for i in 1:r
        xi = x[i]
        for j in i:r
            out[k] = xi * x[j]
            k += 1
        end
    end
    return out
end

"""
    $(TYPEDSIGNATURES)

Allocate and return the unique quadratic monomials of `x`.
"""
function quadratic_monomials(x::AbstractVector{T}) where {T}
    r = length(x)
    out = Vector{float(T)}(undef, (r * (r + 1)) ÷ 2)
    return quadratic_monomials!(out, x)
end

"""
    $(TYPEDSIGNATURES)

Evaluate the reduced right-hand side of `model` at reduced state `xr` with optional
input `u`.
"""
function reduced_dynamics(
        model::OperatorInferenceModel{T},
        xr::AbstractVector,
        u::Union{AbstractVector, Nothing} = nothing
    ) where {T}
    length(xr) == size(model.basis, 2) ||
        throw(DimensionMismatch("reduced state has length $(length(xr)), expected $(size(model.basis, 2))"))
    Tv = promote_type(T, eltype(xr))
    if u !== nothing
        Tv = promote_type(Tv, eltype(u))
    end
    dx = zeros(Tv, length(xr))
    if model.c !== nothing
        dx .+= model.c
    end
    if model.A !== nothing
        mul!(dx, model.A, xr, true, true)
    end
    if model.B !== nothing
        u === nothing &&
            throw(ArgumentError("model has an input operator B but no input u was provided"))
        length(u) == size(model.B, 2) ||
            throw(DimensionMismatch("input has length $(length(u)), expected $(size(model.B, 2))"))
        mul!(dx, model.B, u, true, true)
    end
    if model.H !== nothing
        q = quadratic_monomials(xr)
        mul!(dx, model.H, q, true, true)
    end
    return dx
end

"""
    $(TYPEDSIGNATURES)

Dense POD basis for Operator Inference. Prefer dense SVD over truncated SVD so that
requesting the full numerical rank still yields orthonormal columns.
"""
function _opinf_pod_basis(snapshot::AbstractMatrix{T}, dim::Integer) where {T}
    reducer = POD(snapshot, dim)
    reduce!(reducer, SVD())
    V = reducer.rbasis
    if norm(V' * V - I(size(V, 2))) > sqrt(eps(T)) * size(V, 2)
        throw(ArgumentError("computed POD basis is not orthonormal; try a smaller nmodes"))
    end
    return V
end

function _assert_orthonormal_basis(V::AbstractMatrix{T}; atol = sqrt(eps(T))) where {T}
    r = size(V, 2)
    if norm(V' * V - I(r)) > atol * max(r, 1)
        throw(ArgumentError("basis columns must be orthonormal"))
    end
    return nothing
end

"""
    $(TYPEDSIGNATURES)

Build the Operator Inference data matrix whose columns are the selected operator
features for each snapshot. Snapshot data are stored with time along dimension 2
in `Xr` (`r × k`).
"""
function _opinf_data_matrix(
        Xr::AbstractMatrix{T};
        linear::Bool,
        quadratic::Bool,
        inputs::Union{Nothing, AbstractMatrix{<:Number}},
        constant::Bool
    ) where {T}
    r, k = size(Xr)
    blocks = Matrix{T}[]
    widths = Int[]
    if constant
        push!(blocks, ones(T, k, 1))
        push!(widths, 1)
    end
    if linear
        push!(blocks, Matrix{T}(Xr'))
        push!(widths, r)
    end
    if inputs !== nothing
        U = Matrix{T}(inputs)
        size(U, 2) == k || throw(ArgumentError("inputs must have one column per snapshot"))
        push!(blocks, Matrix{T}(U'))
        push!(widths, size(U, 1))
    end
    if quadratic
        nquad = (r * (r + 1)) ÷ 2
        Q = Matrix{T}(undef, k, nquad)
        x = Vector{T}(undef, r)
        q = Vector{T}(undef, nquad)
        for j in 1:k
            @inbounds for i in 1:r
                x[i] = Xr[i, j]
            end
            quadratic_monomials!(q, x)
            @inbounds for i in 1:nquad
                Q[j, i] = q[i]
            end
        end
        push!(blocks, Q)
        push!(widths, nquad)
    end
    isempty(blocks) &&
        throw(ArgumentError("at least one of linear, quadratic, inputs, or constant must be enabled"))
    return hcat(blocks...), widths
end

function _unpack_opinf_operators(
        O::AbstractMatrix{T},
        widths::Vector{Int};
        linear::Bool,
        quadratic::Bool,
        has_inputs::Bool,
        constant::Bool
    ) where {T}
    offset = 0
    idx = 1
    c = nothing
    A = nothing
    B = nothing
    H = nothing
    if constant
        w = widths[idx]
        idx += 1
        c = vec(O[:, (offset + 1):(offset + w)])
        offset += w
    end
    if linear
        w = widths[idx]
        idx += 1
        A = O[:, (offset + 1):(offset + w)]
        offset += w
    end
    if has_inputs
        w = widths[idx]
        idx += 1
        B = O[:, (offset + 1):(offset + w)]
        offset += w
    end
    if quadratic
        w = widths[idx]
        H = O[:, (offset + 1):(offset + w)]
        offset += w
    end
    offset == size(O, 2) || throw(ArgumentError("internal operator packing mismatch"))
    return A, H, B, c
end

function _normalize_inputs(inputs, k::Int, ::Type{T}) where {T}
    inputs === nothing && return nothing
    if inputs isa AbstractVector
        length(inputs) == k ||
            throw(ArgumentError("vector inputs must have one entry per snapshot"))
        return reshape(Vector{T}(inputs), 1, k)
    end
    U = Matrix{T}(inputs)
    size(U, 2) == k || throw(ArgumentError("inputs must have one column per snapshot"))
    return U
end

"""
    opinf(X, Xdot; nmodes, basis = nothing, linear = true, quadratic = false,
          inputs = nothing, constant = false, λ = 0) -> OperatorInferenceModel

Learn a polynomial reduced model by Operator Inference.

Snapshot matrix `X` and derivative matrix `Xdot` store one snapshot per column
(state dimension × number of snapshots). A POD basis with `nmodes` columns is
computed from `X` unless `basis` is provided. The projected least-squares problem
```math
\\min_O \\| D O^\\top - \\dot X_r^\\top \\|_F^2 + \\lambda \\|O\\|_F^2
```
is solved for the selected operator blocks.

# Arguments
- `X::AbstractMatrix`: state snapshots as columns.
- `Xdot::AbstractMatrix`: time-derivative snapshots as columns, conforming to `X`.

# Keywords
- `nmodes::Integer`: number of POD modes. Required when `basis` is omitted.
- `basis = nothing`: optional orthonormal basis ``V``. When provided, `nmodes` defaults
  to `size(basis, 2)`.
- `linear = true`: infer a linear operator ``A``.
- `quadratic = false`: infer a quadratic operator ``H`` on unique monomials.
- `inputs = nothing`: optional input snapshots as an `(input dimension × snapshots)`
  matrix, or a length-`snapshots` vector for a single input.
- `constant = false`: infer a constant forcing term ``c``.
- `λ = 0`: Tikhonov regularization weight applied isotropically to all operator
  coefficients. When positive, the ridge problem is solved via an augmented QR
  factorization.

# Returns
- [`OperatorInferenceModel`](@ref): inferred basis and operator blocks.

# Throws
- `ArgumentError` / `DimensionMismatch`: if sizes are inconsistent, the basis is not
  orthonormal, or no operator block is requested.

# Examples
```jldoctest
julia> using ModelOrderReduction, LinearAlgebra

julia> Atrue = [-1.0 0.0; 0.0 -2.0];

julia> X = reduce(hcat, [[exp(-t), exp(-2t)] for t in 0:0.05:3]);

julia> Xdot = Atrue * X;

julia> model = opinf(X, Xdot; basis = Matrix{Float64}(I, 2, 2));

julia> model.A ≈ Atrue
true
```

# References
- Peherstorfer, B. & Willcox, K. (2016). Data-driven operator inference for
  nonintrusive projection-based model reduction. *CMAME*.
- Qian, E., Kramer, B., Peherstorfer, B. & Willcox, K. (2020). Lift & Learn.
"""
function opinf(
        X::AbstractMatrix,
        Xdot::AbstractMatrix;
        nmodes::Union{Integer, Nothing} = nothing,
        basis::Union{AbstractMatrix, Nothing} = nothing,
        linear::Bool = true,
        quadratic::Bool = false,
        inputs = nothing,
        constant::Bool = false,
        λ::Real = 0
    )
    size(X) == size(Xdot) || throw(DimensionMismatch("X and Xdot must have the same size"))
    T = float(promote_type(eltype(X), eltype(Xdot)))
    T <: AbstractFloat ||
        throw(ArgumentError("opinf currently supports real floating-point snapshots only"))
    X = Matrix{T}(X)
    Xdot = Matrix{T}(Xdot)
    nstate, k = size(X)
    nstate >= 1 || throw(ArgumentError("state dimension must be positive"))
    k >= 1 || throw(ArgumentError("need at least one snapshot"))
    λ < 0 && throw(ArgumentError("regularization weight λ must be nonnegative"))

    if basis === nothing
        nmodes isa Integer ||
            throw(ArgumentError("nmodes is required when basis is not provided"))
        nmodes = Int(nmodes)
        (0 < nmodes <= min(nstate, k)) ||
            throw(ArgumentError("nmodes must be in 1:min(state dimension, snapshots)"))
        V = _opinf_pod_basis(X, nmodes)
    else
        V = Matrix{T}(basis)
        size(V, 1) == nstate || throw(DimensionMismatch("basis must have $nstate rows"))
        nmodes = nmodes === nothing ? size(V, 2) : Int(nmodes)
        size(V, 2) == nmodes ||
            throw(ArgumentError("nmodes must equal the number of basis columns"))
        nmodes >= 1 || throw(ArgumentError("basis must have at least one column"))
        _assert_orthonormal_basis(V)
    end

    Umat = _normalize_inputs(inputs, k, T)
    Xr = V' * X
    Rr = V' * Xdot
    D, widths = _opinf_data_matrix(
        Xr;
        linear = linear,
        quadratic = quadratic,
        inputs = Umat,
        constant = constant
    )
    # D is k × p, Rr' is k × r; solve D * O' ≈ Rr' in the least-squares sense.
    Rt = Matrix{T}(Rr')
    p = size(D, 2)
    if size(D, 1) < p && λ == 0
        @warn "Operator Inference least-squares problem is underdetermined (snapshots < features); consider more data, fewer terms, or λ > 0" snapshots = size(
            D, 1
        ) features = p
    end
    if λ == 0
        Ot = qr(D, ColumnNorm()) \ Rt
    else
        √λ = sqrt(T(λ))
        Dλ = vcat(D, √λ * I(p))
        Rtλ = vcat(Rt, zeros(T, p, size(Rt, 2)))
        Ot = qr(Dλ, ColumnNorm()) \ Rtλ
    end
    O = Matrix{T}(Ot')
    A, H, B, c = _unpack_opinf_operators(
        O,
        widths;
        linear = linear,
        quadratic = quadratic,
        has_inputs = Umat !== nothing,
        constant = constant
    )
    return OperatorInferenceModel{T}(V, A, H, B, c)
end

"""
    $(TYPEDSIGNATURES)

Operator Inference from a vector-of-snapshots representation, matching the
[`POD`](@ref) snapshot convention.
"""
function opinf(
        X::AbstractVector{<:AbstractVector},
        Xdot::AbstractVector{<:AbstractVector};
        kwargs...
    )
    Xm = reduce(hcat, X)
    Xdm = reduce(hcat, Xdot)
    return opinf(Xm, Xdm; kwargs...)
end
