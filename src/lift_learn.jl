using SymbolicUtils, Symbolics, ModelingToolkit, Bijections

function polynomialization(sys::ODESystem)
    deqs = ModelingToolkit.get_eqs(sys)
    rhs = Symbolics.rhss(deqs)
    lhs = Symbolics.lhss(deqs)
    dvs = map(only ∘ arguments, lhs) # TODO
    iv = ModelingToolkit.get_iv(sys)
    D = Differential(iv)

    deqs = Equation[]
    new_vars = Term[]
    transformation = Bijection()

    function create_var()
        new_var_i += 1
        new_var = Symbolics.variable(:y, new_var_i;
                                     T = SymbolicUtils.FnType{Tuple, Real})(iv)
        push!(new_vars, new_var)
        return new_var
    end
    new_var_i = 0

    while true
        coeff1, coeff2, monomials2, residuals = semiquadratic_form(rhs, dvs)
        # TODO mocking test for semiquadratic_form

        if iszero(residuals)
            append!(deqs, lhs .~ rhs)
            break
        end

        for (m_i, monomial) in enumerate(monomials2)
            if !iszero(monomial)
                monomials2[m_i] = get!(create_var, transformation, monomial)
            end
        end

        for (eq_i, residual) in enumerate(residuals)
            if !Symbolics.iszero(residual)
                val = Symbolics.unwrap(residual)
                if SymbolicUtils.isadd(val)
                    new_residual = Dict()
                    for (term, coeff) in val.dict
                        var = get!(create_var, transformation, term)
                        new_residual[var] = coeff
                    end
                    residuals[eq_i] = SymbolicUtils.Add(Real, 0, new_residual)
                elseif SymbolicUtils.ismul(val)
                    # TODO
                    term = SymbolicUtils.Mul(Real, 1, val.dict)
                    var = get!(create_var, transformation, term)
                    residuals[eq_i] = val.coeff * var
                elseif SymbolicUtils.isterm(val)
                    var = get!(create_var, transformation, val)
                    residuals[eq_i] = var
                elseif SymbolicUtils.ispow(val)
                    # TODO
                else
                    # TODO Div, PolyForm
                end
            end
        end
        rhs = coeff1 * dvs + coeff2 * monomials2 + residuals
        append!(deqs, lhs .~ rhs)

        g = transformation.(new_vars)
        gₓ = Symbolics.sparsejacobian(g, dvs)
        sub_gₓ = substitute.(Symbolics.unwrap.(gₓ), (transformation,))
        rhs = sub_gₓ * Symbolics.rhss(deqs)

        lhs = D.(new_vars)
        append!(dvs, new_vars)
        empty!(new_vars)
    end

    # TODO
    ODESystem(deqs, iv, dvs, parameters(sys);
              name = Symbol(nameof(sys), "_polynomialized"))
end
