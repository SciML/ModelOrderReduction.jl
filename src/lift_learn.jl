using SymbolicUtils, Symbolics, ModelingToolkit, Bijections

function polynomialization(sys::ODESystem)
    deqs = ModelingToolkit.get_eqs(sys)
    dvs = ModelingToolkit.get_states(sys)
    iv = ModelingToolkit.get_iv(sys)
    D = Differential(iv)
    rhs = Symbolics.rhss(deqs)

    new_deqs = Equation[]
    new_dvs = copy(dvs)
    new_vars = Term[]
    transformation = Bijection()

    function create_var()
        new_var_i += 1
        new_var = Symbolics.variable(:y, new_var_i; T = SymbolicUtils.FnType{Tuple, Real})(iv)
        push!(new_vars, new_var)
        return new_var
    end
    new_var_i = 0

    coeff1, coeff2, monomials2, residual = semiquadratic_form(rhs, new_dvs)
    for (eq_i, (deq, nonquadratic)) in enumerate(zip(deqs, residual))
        if !Symbolics.iszero(nonquadratic)
            if SymbolicUtils.isadd(nonquadratic.val)
                new_residual = Dict()
                for (term, coeff) in nonquadratic.val.dict
                    var = get!(create_var, transformation, term)
                    new_residual[var] = coeff
                end
                residual[eq_i] = SymbolicUtils.Add(Real, 0, new_residual)
            elseif SymbolicUtils.ismul(nonquadratic.val)
                term = SymbolicUtils.Mul(Real, 1, nonquadratic.val.dict)
                var = get!(create_var, transformation, term)
                residual[eq_i] = nonquadratic.val.coeff * var
            elseif nonquadratic.val isa Term
                var = get!(create_var, transformation, nonquadratic.val)
                residual[eq_i] = var
            else
                # TODO
            end
        end
    end
    lhs = Symbolics.lhss(deqs)
    lhs_x = map(only ∘ arguments, lhs)
    rhs = coeff1 * new_dvs + coeff2 * monomials2 + residual
    append!(new_deqs, lhs .~ rhs)

    g = transformation.(new_vars)
    gₓ = Symbolics.sparsejacobian(g, lhs_x)
    sub_gₓ = substitute.(gₓ, (transformation,))
    rhs = sub_gₓ * rhs

    append!(new_dvs, new_vars)

    coeff1, coeff2, monomials2, residual = semiquadratic_form(rhs, new_dvs)
    # TODO: edge cases from semiquadratic_form

    # TODO: while loop
    if iszero(residual)
        lhs = D.(new_vars)
        append!(new_deqs, lhs .~ rhs)
        empty!(new_vars)
    end
    # TODO: variable transformation for new system construction
    ODESystem(new_deqs, iv, new_dvs, parameters(sys);
              name = Symbol(nameof(sys), "_polynomialized"))
end
