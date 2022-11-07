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
    degrees = deg.(ops)
    ind = grlex(degrees)
    measures = [op.measure for op in ops]
    w(t) = prod(m.w(t) for m in measures)
    measure = ProductMeasure(w, measures)
    TensorProductOrthoPoly(degrees, ind, measure, Vector(ops))
end

PolyChaos.dim(tpop::TensorProductOrthoPoly) = size(tpop.ind, 2)

function PolyChaos.computeTensorizedSP(dim::Integer, tpop::TensorProductOrthoPoly)
    if any(op.quad isa EmptyQuad for op in tpop.uni)
        throw(InconsistencyError("at least one quadrature rule missing"))
    end
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
- `parameters`: Independent random variables ``\\mathbf X``.
- `uni_basis`: Univariate orthogonal polynomial basis ``\\{ψ_{α_i}^{(i)}(x_i)\\}_{α_i=0}^{r_i}`` for each ``X_i``.
"""
function PCE(states::AbstractVector{Num}, parameters::AbstractVector{Num},
             uni_basis::AbstractVector{<:AbstractOrthoPoly})
    states = Symbolics.scalarize(states)
    parameters = Symbolics.scalarize(parameters)
    tensor_basis = TensorProductOrthoPoly(uni_basis)
    moments = [Symbolics.variable(Symbol(:C, i), view(tensor_basis.ind, :, j)...;
                                  T = Symbolics.FnType)
               for j in axes(tensor_basis.ind, 2), i in eachindex(states)]
    PCE(states, parameters, tensor_basis, moments)
end
