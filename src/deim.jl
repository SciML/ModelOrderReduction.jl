"""
$(TYPEDSIGNATURES)

Compute the DEIM interpolation indices for the given projection basis.

The orthonormal `basis` should not be a sparse matrix.
"""
function deim_interpolation_indices(basis::AbstractMatrix)::Vector{Int}
    dim = size(basis, 2)
    indices = Vector{Int}(undef, dim)
    @inbounds @views begin
        r = abs.(basis[:, 1])
        indices[1] = argmax(r)
        for l in 2:dim
            U = basis[:, 1:(l - 1)]
            P = indices[1:(l - 1)]
            PᵀU = U[P, :]
            uₗ = basis[:, l]
            Pᵀuₗ = uₗ[P, :]
            c = vec(PᵀU \ Pᵀuₗ)
            mul!(r, U, c)
            @. r = abs(uₗ - r)
            indices[l] = argmax(r)
        end
    end
    return indices
end

function qdeim_interpolation_indices(basis::AbstractMatrix)::Vector{Int}
    dim = size(basis, 2)
    return qr(basis', ColumnNorm()).p[1:dim]
end

function odeim_interpolation_indices(basis::AbstractMatrix, sampling_dim::Int)::Vector{Int}
    dim = size(basis, 2)
    @assert sampling_dim >= dim && sampling_dim <= size(basis,1) "Invalid sampling dimension"

    # Compute the first dim points with QDEIM
    p = qdeim_interpolation_indices(basis)

    # select points n, ..., m
    for _ in (length(p) + 1):m
        F = svd(p' * basis)
        S = F.S
        W = F.V
        gap = S[end - 1]^2 - S[end]^2  # eigengap
        proj_basis = transpose(W) * basis
        r = gap .+ sum(proj_basis.^2, dims=1)
        r .= r .- sqrt.((gap + sum(proj_basis.^2, dims=1)).^2 .- 4 * gap * proj_basis[end, :].^2)
        indices = sortperm(r, rev=true)
        e = 1
        while any(indices[e] .== p)
            e += 1
        end
        push!(p, indices[e])
    end
    return p
end

"""
$(SIGNATURES)

Compute the reduced model by applying the Discrete Empirical Interpolation Method (DEIM).

This method allows users to input the projection matrices of their choice.

Given the projection matrix ``V\\in\\mathbb R^{n\\times k}`` for the dependent variables
``\\mathbf y\\in\\mathbb R^n`` and the projection matrix
``U\\in\\mathbb R^{n\\times m}`` for the nonlinear function ``\\mathbf F\\in\\mathbb R^n``,
the full-order model (FOM)
```math
\\frac{d}{dt}\\mathbf y(t)=A\\mathbf y(t)+\\mathbf g(t)+\\mathbf F(\\mathbf y(t))
```
is transformed to the reduced-order model (ROM)
```math
\\frac{d}{dt}\\hat{\\mathbf y}(t)=\\underbrace{V^TAV}_{k\\times k}\\hat{\\mathbf y}(t)+V^T
\\mathbf g(t)+\\underbrace{V^TU(P^TU)^{-1}}_{k\\times m}\\underbrace{P^T\\mathbf F(V
\\hat{\\mathbf y}(t))}_{m\\times1}
```
where ``P=[\\mathbf e_{\\rho_1},\\dots,\\mathbf e_{\\rho_m}]\\in\\mathbb R^{n\\times m}``,
``\\rho_1,\\dots,\\rho_m`` are interpolation indices from the DEIM point selection
algorithm, and ``\\mathbf e_{\\rho_i}=[0,\\ldots,0,1,0,\\ldots,0]^T\\in\\mathbb R^n`` is
the ``\\rho_i``-th column of the identity matrix ``I_n\\in\\mathbb R^{n\\times n}``.

Besides the standard DEIM algorithm for interpolation, this method also supports the QDEIM
and the ODEIM algorithms. The ODEIM algorithm requires an additional parameter `odeim_dim`
to specify the number of the oversampled interpolation points.

# Arguments
- `full_vars::AbstractVector`: the dependent variables ``\\underset{n\\times 1}{\\mathbf y}`` in FOM.
- `linear_coeffs::AbstractMatrix`: the coefficient matrix ``\\underset{n\\times n}A`` of linear terms in FOM.
- `constant_part::AbstractVector`: the constant terms ``\\underset{n\\times 1}{\\mathbf g}`` in FOM.
- `nonlinear_part::AbstractVector`: the nonlinear functions ``\\underset{n\\times 1}{\\mathbf F}`` in FOM.
- `reduced_vars::AbstractVector`: the dependent variables ``\\underset{k\\times 1}{\\hat{\\mathbf y}}`` in the reduced-order model.
- `linear_projection_matrix::AbstractMatrix`: the projection matrix ``\\underset{n\\times k}V`` for the dependent variables ``\\mathbf y``.
- `nonlinear_projection_matrix::AbstractMatrix`: the projection matrix ``\\underset{n\\times m}U`` for the nonlinear functions ``\\mathbf F``.
- `interpolation_algo::Symbol`: the interpolation algorithm, which can be `:deim`, `:qdeim`, or `:odeim`.

# Return
- `reduced_rhss`: the right-hand side of ROM.
- `linear_projection_eqs`: the linear projection mapping ``\\mathbf y=V\\hat{\\mathbf y}``.

# References
- [DEIM](https://epubs.siam.org/doi/abs/10.1137/110822724): Chaturantabut and Sorensen, 2012.
- [QDEIM](http://epubs.siam.org/doi/10.1137/15M1019271): Drmac and Gugercin, 2016.
- [ODEIM](https://epubs.siam.org/doi/10.1137/19M1307391): Peherstorfer, Drmac, and Gugercin, 2020.
"""
function deim(full_vars::AbstractVector, linear_coeffs::AbstractMatrix,
              constant_part::AbstractVector, nonlinear_part::AbstractVector,
              reduced_vars::AbstractVector, linear_projection_matrix::AbstractMatrix,
              nonlinear_projection_matrix::AbstractMatrix,
              interpolation_algo::Symbol; odeim_dim::Integer, kwargs...)
    # rename variables for convenience
    y = full_vars
    A = linear_coeffs
    g = constant_part
    F = nonlinear_part
    ŷ = reduced_vars
    V = linear_projection_matrix
    U = nonlinear_projection_matrix

    # original vars to reduced vars
    linear_projection_eqs = Symbolics.scalarize(y .~ V * ŷ)
    linear_projection_dict = Dict(eq.lhs => eq.rhs for eq in linear_projection_eqs)

    if interpolation_algo == :deim
        indices = deim_interpolation_indices(U) # DEIM interpolation indices
    elseif interpolation_algo == :qdeim
        indices = qdeim_interpolation_indices(U) # QDEIM interpolation indices
    elseif interpolation_algo == :odeim
        indices = odeim_interpolation_indices(U, odeim_dim) # ODEIM interpolation indices
    end
    # the DEIM projector (not DEIM basis) satisfies
    # F(original_vars) ≈ projector * F(pod_basis * reduced_vars)[indices]
    projector = ((@view U[indices, :])' \ (U' * V))'
    temp = substitute.(F[indices], (linear_projection_dict,); kwargs...)
    F̂ = projector * temp # DEIM approximation for nonlinear func F

    Â = V' * A * V
    ĝ = V' * g
    reduced_rhss = Â * ŷ + ĝ + F̂
    return reduced_rhss, linear_projection_eqs
end

"""
    $(FUNCTIONNAME)(
        sys::ModelingToolkit.ODESystem,
        snapshot::AbstractMatrix,
        pod_dim::Integer;
        deim_dim::Integer = pod_dim,
        name::Symbol = Symbol(nameof(sys), :_deim),
        interpolation_algo::Symbol = :deim,
        kwargs...
    ) -> ModelingToolkit.ODESystem

Reduce a `ModelingToolkit.ODESystem` using the Proper Orthogonal Decomposition (POD) with
the Discrete Empirical Interpolation Method (DEIM).

`snapshot` should be a matrix with the data of each time instance as a column.

The LHS of equations in `sys` are all assumed to be 1st order derivatives. Use
`ModelingToolkit.ode_order_lowering` to transform higher order ODEs before applying DEIM.

`sys` is assumed to have no internal systems. End users are encouraged to call
`ModelingToolkit.structural_simplify` beforehand.

The POD basis used for DEIM interpolation is obtained from the snapshot matrix of the
nonlinear terms, which is computed by executing the runtime-generated function for
nonlinear expressions. 

Additional to the DEIM algorithm, this function also supports the QDEIM and ODEIM. For ODEIM,
the `odeim_dim` parameter specifies the number of oversampled interpolation points.
"""
function deim(sys::ODESystem, snapshot::AbstractMatrix, pod_dim::Integer;
              deim_dim::Integer = pod_dim, odeim_dim::Integer = pod_dim,
              name::Symbol = Symbol(nameof(sys), :_deim),
              interpolation_algo::Symbol = :deim, kwargs...)::ODESystem
    @assert interpolation_algo ∈ (:deim, :qdeim, :odeim) "Invalid interpolation algorithm"
    sys = deepcopy(sys)
    @set! sys.name = name

    # handle ODESystem.substitutions
    # https://github.com/SciML/ModelingToolkit.jl/issues/1754
    sys = tearing_substitution(sys; kwargs...)

    iv = ModelingToolkit.get_iv(sys) # the single independent variable
    D = Differential(iv)
    dvs = ModelingToolkit.get_states(sys) # dependent variables

    pod_reducer = POD(snapshot, pod_dim)
    reduce!(pod_reducer, TSVD())
    V = pod_reducer.rbasis # POD basis

    var_name = gensym(:ŷ)
    ŷ = (@variables $var_name(iv)[1:pod_dim])[1]
    @set! sys.states = Symbolics.value.(Symbolics.scalarize(ŷ)) # new variables from POD
    ModelingToolkit.get_var_to_name(sys)[Symbolics.getname(ŷ)] = Symbolics.unwrap(ŷ)

    deqs, eqs = get_deqs(sys) # split eqs into differential and non-differential equations
    rhs = Symbolics.rhss(deqs)
    # a sparse matrix of coefficients for the linear part,
    # a vector of constant terms and a vector of nonlinear terms about dvs
    A, g, F = linear_terms(rhs, dvs)

    # generate an in-place function from the symbolic expression of the nonlinear functions
    F_func! = build_function(F, dvs; expression = Val{false}, kwargs...)[2]
    nonlinear_snapshot = similar(snapshot) # snapshot matrix of nonlinear terms
    for i in 1:size(snapshot, 2) # iterate through time instances
        F_func!(view(nonlinear_snapshot, :, i), view(snapshot, :, i))
    end

    deim_reducer = POD(nonlinear_snapshot, deim_dim)
    reduce!(deim_reducer, TSVD())
    U = deim_reducer.rbasis # DEIM projection basis

    reduced_rhss, linear_projection_eqs = deim(dvs, A, g, F, ŷ, V, U, interpolation_algo; odeim_dim, kwargs...)

    reduced_deqs = D.(ŷ) ~ reduced_rhss
    @set! sys.eqs = [Symbolics.scalarize(reduced_deqs); eqs]

    old_observed = ModelingToolkit.get_observed(sys)
    fullstates = [map(eq -> eq.lhs, old_observed); dvs; ModelingToolkit.get_states(sys)]
    new_observed = [old_observed; linear_projection_eqs]
    new_sorted_observed = ModelingToolkit.topsort_equations(new_observed, fullstates;
                                                            kwargs...)
    @set! sys.observed = new_sorted_observed

    inv_dict = Dict(Symbolics.scalarize(ŷ .=> V' * dvs)) # reduced vars to original vars
    @set! sys.defaults = merge(ModelingToolkit.defaults(sys), inv_dict)
end
