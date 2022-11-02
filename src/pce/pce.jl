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
