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

struct TensorProductOrthoPoly <: AbstractOrthoPoly{ProductMeasure, EmptyQuad{Float64}}
    deg::Vector{Int}
    ind::Matrix{Int}
    measure::ProductMeasure
    uni::Vector
end
function TensorProductOrthoPoly(ops::AbstractVector{T}) where {T <: AbstractOrthoPoly}
    n = length(ops)
    degrees = deg.(ops)
    ind = grevlex(n, 0:maximum(degrees), degrees)
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
