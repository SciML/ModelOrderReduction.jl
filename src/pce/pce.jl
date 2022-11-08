"""
$(TYPEDSIGNATURES)
Returns the number of monomials in the multinomial expansion
``(x_1 + x_2 + ⋯ + x_m)^n``.
"""
multi_indices_size(m::Integer, n::Integer)::Int = binomial(m + n - 1, n)
"""
$(TYPEDSIGNATURES)
Compute the count of solutions ``(a_1, a_2, …, a_m)`` with non-negative integers
``a_i`` such that
```math
\\begin{gather*}
\\operatorname{mn} ≤ a_1 + a_2 + ⋯ + a_m ≤ \\operatorname{mx} \\\\
0 ≤ a_i ≤ r_i \\quad ∀ i
\\end{gather*}
```
The count for equality
```math
a_1 + a_2 + ⋯ + a_m = n
```
by using the inclusion–exclusion principle is
```math
∑_{S⊆\\{1,2,…,m\\}}(-1)^{|S|}\\binom{n+m-1-∑_{i∈S}(r_i+1)}{m-1}
```
where ``\\binom{p}{q}=0`` when ``p < 0``.

The final result is the sum of this formula with ``n`` ranging from `mn` to `mx`.
"""
function multi_indices_size(r::AbstractVector{<:Integer}, mn::Integer, mx::Integer)::Int
    m = length(r)
    m₁ = m - 1
    sum(powerset(1:m); init = 0) do S
        l = length(S)
        temp = m₁ - l - sum(@view r[S]; init = 0)
        res = sum(binomial(a, m₁) for a in (mx + temp):-1:max(1, mn + temp); init = 0)
        iseven(l) ? res : -res
    end
end
"""
$(TYPEDSIGNATURES)
Compute the count of solutions ``(a_1, a_2, …, a_m)`` with non-negative integers
``a_i`` such that
```math
\\begin{gather*}
a_1 + a_2 + ⋯ + a_m ≤ \\max_i a_i \\\\
0 ≤ a_i ≤ r_i \\quad ∀ i
\\end{gather*}
```
"""
function multi_indices_size(r::AbstractVector{<:Integer})::Int
    m = length(r)
    mn, mx = extrema(r)
    res = sum(multi_indices_size(m, n) for n in 0:mn)
    if mn == mx
        return res
    end
    res + multi_indices_size(r, mn + 1, mx)
end

"""
$(TYPEDSIGNATURES)
Return a matrix where the columns are the degrees ``(d_1, d_2, …, d_m)`` of monomials
``x_1^{d_1}x_2^{d_2}⋯x_m^{d_m}`` in the graded lexicographic order such that
```math
\\begin{gather*}
d_1 + d_2 + ⋯ + d_m ≤ \\max_i r_i \\\\
0 ≤ d_i ≤ r_i \\quad ∀ i
\\end{gather*}
```

As `$(FUNCTIONNAME)` has exponential space complexity, [`multi_indices_size`](@ref) is used
to compute the matrix size in order to reduce the number of allocations.
"""
function grlex(r::AbstractVector{<:Integer})::Matrix{Int}
    mn, mx = extrema(r)
    indices_size = multi_indices_size(r)
    n_term = length(r)
    res = zeros(Int, n_term, indices_size)
    indices_i = 2
    @inbounds for total_degree in 1:mn
        for stars in combinations(1:(n_term + total_degree - 1), total_degree)
            degree = @view res[:, indices_i]
            for (i, s) in enumerate(stars)
                degree[s - i + 1] += 1
            end
            indices_i += 1
        end
    end
    @inbounds for total_degree in (mn + 1):mx
        for stars in combinations(1:(n_term + total_degree - 1), total_degree)
            degree = @view res[:, indices_i]
            for (i, s) in enumerate(stars)
                degree[s - i + 1] += 1
            end
            if any(degree .> r)
                degree .= 0
                continue
            end
            indices_i += 1
            if indices_i > indices_size
                return res
            end
        end
    end
    res
end

"""
$(TYPEDEF)
`$(FUNCTIONNAME)` represents multivariate orthogonal polynomial bases formed as the tensor
product of univariate `PolyChaos.AbstractOrthoPoly`s.

`$(FUNCTIONNAME)` is similar to `PolyChaos.MultiOrthoPoly`, but allows different degree
truncation for univariate bases. Multi-indices of `$(FUNCTIONNAME)` are stored in the
columns as Julia is column-major, while multi-indices are rows in the index matrix of
`PolyChaos.MultiOrthoPoly`.

By default, the sum of multi-indices are restricted to the maximum degree among the
univariate bases.

# Fields
$(TYPEDFIELDS)
"""
struct TensorProductOrthoPoly{OP <: Union{AbstractOrthoPoly, AbstractCanonicalOrthoPoly}} <:
       AbstractOrthoPoly{ProductMeasure, AbstractQuad{Float64}}
    "The degree truncation of each univariate orthogonal polynomials."
    deg::Vector{Int}
    "Multi-indices in the columns."
    ind::Matrix{Int}
    "Product measure."
    measure::ProductMeasure
    "Univariate orthogonal polynomials."
    uni::Vector{OP}
end
"""
$(TYPEDSIGNATURES)
"""
function TensorProductOrthoPoly(ops::AbstractVector{
                                                    <:Union{AbstractOrthoPoly,
                                                            AbstractCanonicalOrthoPoly}})
    if any(op.quad isa EmptyQuad for op in ops)
        throw(InconsistencyError("at least one quadrature rule missing"))
    end
    degrees = deg.(ops)
    ind = grlex(degrees)
    measures = [op.measure for op in ops]
    w(t) = prod(m.w(t) for m in measures)
    measure = ProductMeasure(w, measures)
    TensorProductOrthoPoly(degrees, ind, measure, Vector(ops))
end

PolyChaos.dim(tpop::TensorProductOrthoPoly) = size(tpop.ind, 2)

function PolyChaos.computeTensorizedSP(dim::Integer, tpop::TensorProductOrthoPoly)
    computeTensorizedSP(dim, tpop.uni, transpose(tpop.ind))
end

"""
$(TYPEDSIGNATURES)
Construct a `$(FUNCTIONNAME)` which is used to compute and store the results of inner
products
```math
⟨ϕ_{i_1}ϕ_{i_2}⋯ϕ_{i_{m-1}},ϕ_{i_m}⟩
```
where ``m`` is input argument `dim`.
"""
function PolyChaos.Tensor(dim::Int, tpop::TensorProductOrthoPoly)
    tensor_entries = computeTensorizedSP(dim, tpop)
    getfun(ind) = getentry(ind, tensor_entries, transpose(tpop.ind), dim)
    Tensor(dim, tensor_entries, getfun, tpop)
end

"""
$(TYPEDEF)
Suppose a variable ``Y`` with finite variance is a function of ``n`` independent but not
identically distributed random variables ``X_1, …, X_n`` with joint density
``p(x_1, …, x_n) = p_1(x_1) p_2(x_2) ⋯ p_n(x_n)``. Then the Polynomial Chaos Expansion
(PCE) for ``Y = Y(X_i), X_i ∼ π_{x_i}, i = 1, …, n``, takes the form
```math
y(x_1, …, x_n) = ∑_{α_1=0}^∞ ∑_{α_2=0}^∞ ⋯ ∑_{α_n=0}^∞
C_{(α_1, α_2, …, α_n)} Ψ_{(α_1, α_2, …, α_n)}(x_1, …, x_n)
```
Here the summation runs across all possible combinations of the multi-index
``α⃗ = (α_1, …, α_n)``.

The set of multivariate orthogonal polynomials is defined as
```math
Ψ_{α⃗}(x_1, …, x_n) = ∏_{i=1}^n ψ_{α_i}^{(i)}(x_i)
```
where ``\\{ψ_{α_i}^{(i)}(x_i)\\}_{α_i=0}^∞`` is the family of univariate orthogonal
polynomials with respect to ``p_i``.

# Fields
$(TYPEDFIELDS)
"""
struct PCE
    "The random variables ``\\mathbf Y`` that are represented by other random variables."
    states::Vector{Num}
    "The independent random variables ``\\mathbf X``."
    parameters::Vector{Num}
    """
    The tensor-product-based multivariate basis underpinning the PCE, which stores the
    univariate orthogonal polynomial basis ``\\{ψ_{α_i}^{(i)}(x_i)\\}_{α_i=0}^{r_i}`` for
    each ``X_i`` and multi-indices ``α⃗``.
    """
    tensor_basis::TensorProductOrthoPoly
    "The coefficients ``C_{(α_1, α_2, …, α_n)}`` for each ``Y_i`` in the columns."
    moments::Matrix
    "The mapping ``Y => ∑_α C_α Ψ_α`` for each ``Y``."
    ansatz::Dict{Num, Num}
    "Results of tensor inner products ``⟨Ψ_{i_1}Ψ_{i_2}⋯Ψ_{i_{m-1}},Ψ_{i_m}⟩``."
    tensors::Dict{Int, Tensor}
end
"""
$(TYPEDSIGNATURES)
Construct a `$(FUNCTIONNAME)` object.

In practice, the infinite series of multi-indices must be truncated. By default, besides
the upper bound for the degree of each univariate orthogonal polynomial, the total degree
is restricted to the maximum degree among the univariate bases. That is
```math
\\begin{gather*}
α_1 + α_2 + ⋯ α_n ≤ \\max_i r_i \\\\
0 ≤ α_i ≤ r_i \\quad ∀ i
\\end{gather*}
```
where ``r_i`` is the degree upper bound for the univariate basis corresponding to ``X_i``.

# Arguments
- `states`: Random vairables ``\\mathbf Y`` that are represented by other random variables.
- `ivs`: Independent vairables of ``\\mathbf Y``. Enter an empty vector if there is none.
- `parameters`: Independent random variables ``\\mathbf X``.
- `uni_basis`: Univariate orthogonal polynomial basis ``\\{ψ_{α_i}^{(i)}(x_i)\\}_{α_i=0}^{r_i}`` for each ``X_i``.
"""
function PCE(states::AbstractVector{Num}, ivs::AbstractVector{Num},
             parameters::AbstractVector{Num},
             uni_basis::AbstractVector{
                                       <:Union{AbstractOrthoPoly, AbstractCanonicalOrthoPoly
                                               }})
    states = Symbolics.scalarize(states)
    parameters = Symbolics.scalarize(parameters)
    tensor_basis = TensorProductOrthoPoly(uni_basis)
    C = Matrix{Num}(undef, dim(tensor_basis), length(states))
    Ψ = Vector{Num}(undef, dim(tensor_basis))
    snames = Symbolics.tosymbol.(states)
    for i in axes(tensor_basis.ind, 2)
        name = join(Symbolics.map_subscripts.(view(tensor_basis.ind, :, i)), "ˏ")
        Ψname = Symbol(:Ψ, name)
        # record index of multi-index α⃗ in symbolic metadata
        Ψ[i] = first(@variables $Ψname [description = i])
        for (j, sname) in enumerate(snames)
            Cname = Symbol(:C, sname, name)
            C[i, j] = if isempty(ivs)
                first(@variables $Cname [description = i])
            else
                first(@variables $Cname(ivs...) [description = i])
            end
        end
    end
    # X => x₀ + x₁ψ₁(X)
    # where x₀ and x₁ are affince PCE coefficients of X
    # ψ₁(x) = x - α₀ is the first-order monic basis polynomial of X
    # TODO: Use mean and std of the distribution of X to compute x₀ and x₁
    x_dict = Dict(x => op.α[1] + Ψ[i + 1]
                  for (i, (x, op)) in enumerate(zip(parameters, uni_basis)))
    # Y => ∑_α C_α Ψ_α
    y_dict = Dict(y => dot(view(C, :, i), Ψ) for (i, y) in enumerate(states))
    t2 = Tensor(2, tensor_basis)
    tensors = Dict(2 => t2)
    PCE(states, parameters, tensor_basis, C, y_dict, tensors)
end

# extracting the indices of the factors of as basismonomial
function get_basis_indices(mono::Symbolics.Mul)
    basis_indices = Int[]
    for (term, pow) in mono.dict
        append!(basis_indices, (arguments(term)[end] - 1) * ones(Int, pow))
    end
    return basis_indices
end
function get_basis_indices(mono::Symbolics.Term)
    return [arguments(mono)[end] - 1]
end
function get_basis_indices(mono::Symbolics.Pow)
    return (arguments(mono.base)[end] - 1) * ones(Int, mono.exp)
end
function get_basis_indices(mono::Num)
    return get_basis_indices(Symbolics.unwrap(mono))
end
function get_basis_indices(::Val{1})
    return [0]
end

# Compute the an ascending list of `n`-dimensional multi-indices with fixed `grade` (= sum of entries)
# in graded reverse lexicographic order. Constraints on the degrees considered can be incorporated.
function grevlex(n::Int, grade::Int)
    if n == 1
        return reshape([grade], 1, 1)
    end
    if grade == 0
        return zeros(Int, 1, n)
    end
    sub_ind = grevlex(n - 1, grade)
    ind = hcat(sub_ind, zeros(Int, size(sub_ind, 1)))
    for k in 1:grade
        sub_ind = grevlex(n - 1, grade - k)
        ind = vcat(ind, hcat(sub_ind, k * ones(Int, size(sub_ind, 1))))
    end
    return ind
end
function grevlex(n::Int, grades::AbstractVector{Int})
    return reduce(vcat, [grevlex(n, grade) for grade in grades])
end
function grevlex(n::Int, grade::Int, max_degrees::Vector{Int})
    return grevlex(n, grade, [0:d for d in max_degrees])
end
function grevlex(n::Int, grades::AbstractVector{Int}, max_degrees::Vector{Int})
    return reduce(vcat, [grevlex(n, grade, max_degrees) for grade in grades])
end
function grevlex(n::Int, grade::Int, degree_constraints::Vector{<:AbstractVector})
    if n == 1
        return grade in degree_constraints[1] ? reshape([grade], 1, 1) : zeros(Int, 0, 1)
    end
    if grade == 0
        return all(0 in degs for degs in degree_constraints) ? zeros(Int, 1, n) :
               zeros(Int, 0, n)
    end
    filtered_grades = filter(x -> x <= grade, degree_constraints[end])
    sub_ind = grevlex(n - 1, grade - filtered_grades[1], degree_constraints[1:(end - 1)])
    ind = hcat(sub_ind, filtered_grades[1] * ones(Int, size(sub_ind, 1)))
    for k in filtered_grades[2:end]
        sub_ind = grevlex(n - 1, grade - k, degree_constraints[1:(end - 1)])
        ind = vcat(ind, hcat(sub_ind, k * ones(Int, size(sub_ind, 1))))
    end
    return ind
end
function grevlex(n::Int, grades::AbstractVector{Int},
                 degree_constraints::Vector{<:AbstractVector})
    return reduce(vcat, [grevlex(n, grade, degree_constraints) for grade in grades])
end

# 1. apply PCE ansatz
"""
$(TYPEDSIGNATURES)

Generate linear PCEs for the uncertain parameters.
"""
function generate_parameter_pce(pce::PCE)
    par_dim = length(pce.parameters)
    par_pce = Vector{Pair{eltype(pce.parameters), eltype(pce.sym_basis)}}(undef, par_dim)
    for (i, basis) in enumerate(pce.uni_basis)
        p, op = basis
        par_pce[i] = p => pce.sym_basis[i + 1] + op.α[1]
    end
    return par_pce
end

"""
$(TYPEDSIGNATURES)

Substitute parameters shared between a set of symbolic `eqs` and the PCE `pce` for the corresponding linear PCEs.
"""
function substitute_parameters(eqs::AbstractVector, pce::PCE)
    par_pce = generate_parameter_pce(pce)
    subs_eqs = [substitute(eq, par_pce) for eq in eqs]
    return subs_eqs
end

"""
$(TYPEDSIGNATURES)

Substitute PCE Ansatz defined in `pce` into a set of symbolic equations `eqs`.
"""
function substitute_pce_ansatz(eqs::AbstractVector, pce::PCE)
    subs_eqs = [expand(expand(substitute(eq, pce.ansatz))) for eq in eqs]
    return subs_eqs
end

"""
$(TYPEDSIGNATURES)

Apply PCE ansatz defined in `pce` to a given set of symbolic equations `eqs`.
"""
function apply_ansatz(eqs::AbstractVector, pce::PCE)
    return substitute_pce_ansatz(substitute_parameters(eqs, pce), pce)
end

# 2. extract PCE expansion coeffs
"""
$(TYPEDSIGNATURES)

Given a set of symbolic equations `eqs` involving the basis functions of `pce`,
extract monomials of the basis functions and the corresponding coeffiecients.

# Returns
`Vector` of `Dict`s mapping monomial of basis functions to its coefficient in the individual equations.
"""
function extract_basismonomial_coeffs(eqs::AbstractVector, pce::PCE)
    basismonomial_coeffs = [extract_coeffs(eq, pce.sym_basis) for eq in eqs]
    basismonomial_indices = []
    for coeffs in basismonomial_coeffs
        union!(basismonomial_indices,
               [mono => get_basis_indices(mono) for mono in keys(coeffs)])
    end
    return basismonomial_coeffs, basismonomial_indices
end

# 3. compute inner products
"""
$(TYPEDSIGNATURES)

Evaluate scalar products between all basis functions in `pce` and
basis monomials as characterized by `mono_indices`.
"""
function eval_scalar_products(mono_indices, pce::PCE)
    uni_degs = [deg(op) for op in pce.tensor_basis.uni]
    max_degs = uni_degs
    for (mono, id) in mono_indices
        max_degs = max.(max_degs, vec(sum(pce.tensor_basis.ind[id .+ 1, :], dims = 1)))
    end
    quad_deg = max.(ceil.(Int, 0.5 * (max_degs + uni_degs .+ 1)))

    integrators = map((uni, deg) -> bump_degree(uni, deg), pce.tensor_basis.uni, quad_deg)
    scalar_products = Dict()
    for k in 1:dim(pce.tensor_basis)
        scalar_products[k] = Dict(mono => computeSP(vcat(id, k - 1), pce.tensor_basis,
                                                    integrators)
                                  for (mono, id) in mono_indices)
    end
    return scalar_products
end

# 4. Galerkin projection
"""
$(TYPEDSIGNATURES)

perform Galerkin projection of polynomial expressions characterized by `Dict`s mapping
basis monomials to coefficients.
"""
function galerkin_projection(bm_coeffs::Vector{<:Dict}, scalar_products::Dict,
                             pce::PCE)
    projected_eqs = []
    scaling_factors = computeSP2(pce.tensor_basis)
    for i in eachindex(bm_coeffs)
        eqs = []
        for k in 1:dim(pce.tensor_basis)
            push!(eqs,
                  sum(bm_coeffs[i][mono] * scalar_products[k][mono]
                      for mono in keys(bm_coeffs[i])))
        end
        push!(projected_eqs, eqs ./ scaling_factors)
    end
    return projected_eqs
end

# 5. combine everything
"""
$(TYPEDSIGNATURES)

perform Galerkin projection onto the `pce`.
"""
function pce_galerkin(eqs::AbstractVector, pce::PCE)
    expanded_eqs = apply_ansatz(eqs, pce)
    basismono_coeffs, basismono_idcs = extract_basismonomial_coeffs(expanded_eqs, pce)
    scalar_products = eval_scalar_products(basismono_idcs, pce)
    projected_eqs = galerkin_projection(basismono_coeffs, scalar_products, pce)
    return projected_eqs
end

# 6. high-level interface
# 6a. apply pce to explicit ODE
"""
$(TYPEDSIGNATURES)

Generate moment equations of an `ODESystem` from a given `PCE`-Ansatz via Galerkin projection.
"""
function moment_equations(sys::ODESystem, pce::PCE)
    eqs = [eq.rhs for eq in equations(sys)]
    projected_eqs = pce_galerkin(eqs, pce)
    moment_eqs = reduce(vcat, projected_eqs)
    iv = ModelingToolkit.get_iv(sys)
    params = setdiff(parameters(sys), pce.parameters)
    D = Differential(iv)
    moments = reduce(vcat, pce.moments)
    name = Symbol(String(nameof(sys)) * "_pce")
    pce_system = ODESystem([D(moments[i]) ~ moment_eqs[i] for i in eachindex(moments)],
                           iv, moments, params, name = name)

    n_moments = dim(pce.tensor_basis)
    n_states = length(states(sys))
    pce_eval = function (moment_vals, parameter_values)
        shape_state = [moment_vals[(i * n_moments + 1):((i + 1) * n_moments)]
                       for i in 0:(n_states - 1)]
        return pce(shape_state, parameter_values)
    end
    return pce_system, pce_eval
end
