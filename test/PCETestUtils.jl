function isapprox_sym(exprA, exprB, atol = 1e-6)
    exprA = Symbolics.unwrap(exprA)
    exprB = Symbolics.unwrap(exprB)
    return isapprox_sym(exprA, exprB)
end

function isapprox_sym(exprA::TA, exprB::TB,
                      atol = 1e-6) where {TA, TB <: Union{Symbolics.Mul, Symbolics.Add}}
    approx_equal = true
    approx_equal = isapprox(exprA.coeff, exprB.coeff, atol = atol)
    approx_equal = keys(exprA.dict) == keys(exprB.dict)
    if approx_equal
        for subexpr in keys(exprA.dict)
            approx_equal = isapprox(exprA.dict[subexpr], exprB.dict[subexpr], atol = atol)
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
