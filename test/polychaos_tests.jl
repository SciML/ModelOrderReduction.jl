using Test
using ModelOrderReduction, Symbolics, PolyChaos, ModelingToolkit, LinearAlgebra
MO = ModelOrderReduction

# testing extraction of independent variables 
@variables t, z, u(t), v(t)[1:4], w(t,z), x(t,z)[1:4]

@test isequal(MO.get_independent_vars(u), [t])
@test isequal(MO.get_independent_vars(v[1]), [t])
@test isequal(MO.get_independent_vars(v[2]), [t])
@test isequal(MO.get_independent_vars(w),[t,z])
@test isequal(MO.get_independent_vars(x[2]),[t,z])
@test isequal(MO.get_independent_vars(collect(v)), [[t] for i in 1:length(v)])
@test isequal(MO.get_independent_vars(collect(x)), [[t,z] for i in 1:length(v)])

# test equation for throughout:
@parameters a
@variables t, x(t)
D = Differential(t)
test_equation = [D(x) ~ a*x]

# test PCE generation
n = 5
bases = [a => GaussOrthoPoly(n)]
pce = PCE([x], bases)
@test length(pce.moments[1]) == n + 1
@test length(pce.sym_basis) == n + 1
@test isequal(pce.parameters, [a])

# test PCE ansatz application
eq = [eq.rhs for eq in test_equation]
pce_eq = MO.apply_ansatz(eq, pce)[1]
true_eq = expand(pce.sym_basis[2]*dot(pce.moments[1],pce.sym_basis))
@test isequal(pce_eq, true_eq)

# test extraction of monomial coefficients
coeffs = Dict(pce.sym_basis[i]*pce.sym_basis[2] => pce.moments[1][i] for i in 1:n+1)
extracted_coeffs = MO.extract_coeffs(pce_eq, pce.sym_basis)
@test all(isequal(coeffs[mono],extracted_coeffs[mono]) for mono in keys(coeffs))

basis_indices = Dict([pce.sym_basis[i]*pce.sym_basis[2] => ([i-1, 1], [1,i-1]) for i in 1:n+1])
extracted_coeffs, extracted_basis_indices = MO.extract_basismonomial_coeffs([pce_eq], pce)
extracted_basis_indices = Dict(extracted_basis_indices)
test1 = [isequal(basis_indices[mono][1], extracted_basis_indices[mono]) for mono in keys(basis_indices)]
test2 = [isequal(basis_indices[mono][2], extracted_basis_indices[mono]) for mono in keys(basis_indices)]
@test all(test1 + test2 .>= 1)

# test Galerkin projection
moment_eqs = MO.pce_galerkin(eq, pce)

integrator = MO.bump_degree(pce.pc_basis, n+1)
@test integrator.deg == n+1
@test integrator.measure isa typeof(pce.pc_basis.measure)
@test integrator.measure.measures[1] isa typeof(pce.pc_basis.measure.measures[1])

true_moment_eqs = Num[]
for j in 0:n
    mom_eq = 0.0
    for mono in keys(basis_indices)
        ind = basis_indices[mono][2]
        c = computeSP([ind...,j], integrator)
        mom_eq += c*coeffs[mono]
    end
    push!(true_moment_eqs, mom_eq)
end

function isapprox_sym(exprA, exprB, atol = 1e-6)
    exprA = Symbolics.unwrap(exprA)
    exprB = Symbolics.unwrap(exprB)
    approx_equal = false
    if typeof(exprA) == typeof(exprB)
        approx_equal = isapprox_sym(exprA, exprB, atol)
    end
    return approx_equal
end

function isapprox_sym(exprA::T, exprB::T, atol = 1e-6) where T <: Union{Symbolics.Mul, Symbolics.Add}
    approx_equal = true
    approx_equal = isapprox(exprA.coeff, exprB.coeff, atol = atol)
    approx_equal = keys(exprA.dict) == keys(exprB.dict)
    if approx_equal 
        for subexpr in keys(exprA.dict)
            approx_equal = isapprox(exprA.dict[subexpr], exprB.dict[subexpr], atol=atol)
            if !approx_equal
                break
            end
        end
    end
    return approx_equal
end

function isapprox_sym(exprA::Symbolics.Term, exprB::Symbolics.Term, atol = 1e-6)
    return isequal(exprA, exprB)
end

@test all([isapprox_sym(moment_eqs[1][i], true_moment_eqs[i]) for i in eachindex(true_moment_eqs)])