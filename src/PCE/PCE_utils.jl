import PolyChaos.computeSP2

# getting independent variables
function get_independent_vars(var)
    return []
end
function get_independent_vars(var::Symbolics.Term) where T
    if operation(var) isa Symbolics.Sym
        return arguments(var)
    else
        return reduce(vcat, get_independent_vars(arguments(var)))
    end
end
function get_independent_vars(var::Num)
    return get_independent_vars(Symbolics.unwrap(var))
end
function get_independent_vars(vars::AbstractVector)
    return [get_independent_vars(var) for var in vars]
end

# utiltiites to extracting coefficients of a polynomial in monomial basis in variables `vars`
function split_term(term::Symbolics.Mul, vars)
    coeff = term.coeff
    mono = Num(1.0)
    for var in keys(term.dict)
        if var in vars
            mono *= var^term.dict[var]
        else
            coeff *= var^term.dict[var]
        end
    end
    return coeff, mono
end

function split_term(term::Symbolics.Pow, vars)
    if term.base in vars
        return 1.0, term
    else
        return term, 1.0
    end
end

function split_term(term::T, vars) where T <: Union{Symbolics.Term, Symbolics.Sym}
    if term in vars
        return 1.0, term
    else
        return term, 1.0
    end
end 

function extract_coeffs(expr::Symbolics.Add, vars::Set)
    coeffs = Dict()
    if !iszero(expr.coeff)
        coeffs[Val(1)] = expr.coeff
    end
    for term in keys(expr.dict)
        num_coeff = expr.dict[term]
        var_coeff, mono = split_term(term, vars)
        try 
            coeffs[mono] += num_coeff*var_coeff
        catch
            coeffs[mono] = num_coeff*var_coeff
        end
    end 
    return coeffs
end

function extract_coeffs(expr::Symbolics.Mul, vars::Set)
    coeff, mono = split_term(expr, vars)
    return Dict(mono => coeff)
end

extract_coeffs(expr::Num, vars::Set) = extract_coeffs(Symbolics.unwrap.(expr), vars)
extract_coeffs(expr::Num, vars::AbstractArray) = extract_coeffs(Symbolics.unwrap.(expr), vars)
extract_coeffs(expr, vars::AbstractArray{<:Num}) = extract_coeffs(expr, Symbolics.unwrap.(vars))
extract_coeffs(expr, vars::AbstractArray) = extract_coeffs(expr, Set(vars))
extract_coeffs(expr::Number, vars::Set) = Dict(Val(1) => expr)

# extracting the indices of the factors of as basismonomial
function get_basis_indices(mono::Symbolics.Mul)
    basis_indices = Int[]
    for (term, pow) in mono.dict
        append!(basis_indices, (arguments(term)[end] - 1)*ones(Int, pow))
    end
    return basis_indices
end
function get_basis_indices(mono::Symbolics.Term)
    return [arguments(mono)[end] - 1]
end
function get_basis_indices(mono::Symbolics.Pow)
    return (arguments(mono.base)[end] - 1)*ones(Int, mono.exp)
end
function get_basis_indices(mono::Num)
    return get_basis_indices(Symbolics.unwrap(mono))
end
function get_basis_indices(::Val{1})
    return [0]
end

# bumping the degree of a PolyChaos OrthoPoly object up to ensure exact integration
# PR to PolyChaos -> remove unnecessarily restrictive constructors and allow construction from measures
#                 -> also expose number of points used for quadrature generation for general orthogonal polys
#
measure_parameters(m::AbstractMeasure) = []
measure_parameters(m::Measure) = m.pars
measure_parameters(m::JacobiMeasure) = [m.ashapeParameter, m.bshapeParameter]
measure_parameters(m::genLaguerreMeasure) = [m.shapeParameter]
measure_parameters(m::genHermiteMeasure) = [m.muParameter]
measure_parameters(m::MeixnerPollaczekMeasure) = [m.λParameter, m.ϕParameter]
measure_parameters(m::Beta01Measure) = [m.ashapeParameter, m.bshapeParameter]
measure_parameters(m::GammaMeasure) = [m.shapeParameter, m.rateParameter]

recursion_coeffs(m::JacobiMeasure, deg::Int) = rm_jacobi(deg+1, m.ashapeParameter, m.bshapeParameter)
function OrthoPoly(m::JacobiMeasure, deg::Int)
    α, β = recursion_coeffs(m, deg)
    return JacobiOrthoPoly(deg, α, β, m)
end
function bump_degree(op::JacobiOrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return JacobiOrthoPoly(deg, ps...)
end

recursion_coeffs(m::genLaguerreMeasure, deg::Int) = rm_laguerre(deg+1, m.shapeParameter)
function bump_degree(op::genLaguerreOrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return genLaguerreorthoPoly(deg, ps...)
end

recursion_coeffs(m::MeixnerPollaczekMeasure, deg::Int) = rm_meixner_pollaczek(deg+1, m.λParameter, m.ϕParameter)
function bump_degree(op::MeixnerPollaczekOrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return MeixnerPollaczekOrthoPoly(deg, ps...)
end

recursion_coeffs(m::Beta01Measure, deg::Int) = r_scale(1 / beta(m.ashapeParameter, m.bshapeParameter), rm_jacobi01(deg + 1, m.bshapeParameter - 1.0, m.ashapeParameter - 1.0)...)
function bump_degree(op::Beta01OrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return Beta01OrthoPoly(deg, ps...)
end

recursion_coeffs(m::GammaMeasure, deg::Int) = r_scale((m.rateParameter^m.shapeParameter) / gamma(m.shapeParameter), rm_laguerre(deg+1, m.shapeParameter - 1.0)...)
function bump_degree(op::GammaOrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return GammaOrthoPoly(deg, ps...)
end

recursion_coeffs(m::genHermiteMeasure, deg::Int) = rm_hermite(deg+1, m.muParameter)
function bump_degree(op::genHermiteOrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return genHermiteOrthoPoly(deg, ps...)
end

recursion_coeffs(m::HermiteMeasure, deg::Int) = rm_hermite(deg+1)
function bump_degree(op::HermiteOrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return HermiteOrthoPoly(deg, ps...)
end

recursion_coeffs(m::LaguerreMeasure, deg::Int) = rm_laguerre(deg+1)
function bump_degree(op::LaguerreOrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return LaguerreOrthoPoly(deg, ps...)
end

recursion_coeffs(m::Uniform01Measure, deg::Int) = r_scale(1.0, rm_legendre01(deg+1)...)
function bump_degree(op::Uniform01OrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return Uniform01OrthoPoly(deg, ps...)
end

recursion_coeffs(m::Uniform_11Measure, deg::Int) = r_scale(0.5, rm_legendre(deg+1)...)
function bump_degree(op::Uniform_11OrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return Uniform_11OrthoPoly(deg, ps...)
end

recursion_coeffs(m::GaussMeasure, deg::Int) = r_scale(1/sqrt(2π), rm_hermite_prob(deg+1)...)
function bump_degree(op::GaussOrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return GaussOrthoPoly(deg, ps...)
end

recursion_coeffs(m::LegendreMeasure, deg::Int) = rm_legendre(deg+1)
function bump_degree(op::LegendreOrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return LegendreOrthoPoly(deg, ps...)
end

recursion_coeffs(m::LogisticMeasure, deg::Int) = r_scale(1.0, rm_logistic(deg+1)...)
function bump_degree(op::LogisticOrthoPoly, deg::Int)
    ps = measure_parameters(op.measure)
    return LogisticOrthoPoly(deg, ps...)
end

function bump_degree(op::MultiOrthoPoly, deg::Int)
    return MultiOrthoPoly(bump_degree.(op.uni, deg), deg)
end

# extending computeSP2 for multivariate orthogonal polys
function computeSP2(pc::MultiOrthoPoly)
    n = length(pc.uni)
    m = dim(pc)
    uni_SP2 = [computeSP2(op) for op in pc.uni]
    multi_SP2 = [prod(uni_SP2[j][pc.ind[i,j]+1] for j in 1:n) for i in 1:m]
    return multi_SP2
end