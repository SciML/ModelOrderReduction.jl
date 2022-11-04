"""
$(TYPEDSIGNATURES)
Returns the number of monomials in the multinomial expansion
``(x_1 + x_2 + \\dotsb + x_m)^n``.
"""
multi_indices_size(m::Integer, n::Integer)::Int = binomial(m + n - 1, n)
"""
$(TYPEDSIGNATURES)
Compute the count of solutions ``(a_1, a_2, \\dotsc, a_m)`` with non-negative integers
``a_i`` such that
```math
\\begin{gather*}
\\operatorname{mn} \\leq a_1 + a_2 + \\dotsb + a_m \\leq \\operatorname{mx} \\\\
0 \\leq a_i \\leq r_i \\quad \\forall i
\\end{gather*}
```
The count for equality
```math
a_1 + a_2 + \\dotsb + a_m = n
```
by using the inclusion–exclusion principle is
```math
\\sum_{S\\subseteq\\{1,2,\\dotsc,m\\}}(-1)^{|S|}\\binom{n+m-1-\\sum_{i\\in S}(r_i+1)}{m-1}
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
Compute the count of solutions ``(a_1, a_2, \\dotsc, a_m)`` with non-negative integers
``a_i`` such that
```math
\\begin{gather*}
a_1 + a_2 + \\dotsb + a_m \\leq \\max_i a_i \\\\
0 \\leq a_i \\leq r_i \\quad \\forall i
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
Return a matrix where the columns are the degrees ``(d_1, d_2, \\dotsc, d_m)`` of monomials
``x_1^{d_1}x_2^{d_2}\\dotsm x_m^{d_m}`` in the graded lexicographic order such that
```math
\\begin{gather*}
d_1 + d_2 + \\dotsb + d_m \\leq \\max_i r_i \\\\
0 \\leq d_i \\leq r_i \\quad \\forall i
\\end{gather*}
```
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

struct TensorProductOrthoPoly <: AbstractOrthoPoly{ProductMeasure, EmptyQuad{Float64}}
    deg::Vector{Int}
    ind::Matrix{Int}
    measure::ProductMeasure
    uni::Vector
end
function TensorProductOrthoPoly(ops::AbstractVector{T}) where {T <: AbstractOrthoPoly}
    degrees = deg.(ops)
    ind = grlex(degrees)
    measures = [op.measure for op in ops]
    w(t) = prod(m.w(t) for m in measures)
    measure = ProductMeasure(w, measures)
    TensorProductOrthoPoly(degrees, ind, measure, ops)
end

"""
$(TYPEDEF)

# Fields
$(TYPEDFIELDS)
"""
struct PCE{P <: AbstractOrthoPoly}
    "Independent symbolic random variables ``X``."
    x::Vector{Num}
    "Univariate orthogonal polynomial basis for each ``X_i``."
    basis::Vector{P}
    "The array of multi-indices which could be a sparse grid."
    ind::Matrix{Int}
end
