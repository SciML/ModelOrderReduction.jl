using SymbolicUtils, Symbolics, ModelingToolkit, Bijections

function polynomialization(sys::ODESystem)::ODESystem
    # the input sys may include substitutions that have not been applied,
    # but this is an intrusive method and we need to access all operators,
    # so here we call `full_equations` instead of `equations`
    deqs = full_equations(sys)
    rhs = Symbolics.rhss(deqs)
    lhs = Symbolics.lhss(deqs)
    dvs = map(only ∘ arguments, lhs) # assume all LHS are first order derivatives

    dvs_set = Set(dvs)

    iv = ModelingToolkit.get_iv(sys)
    D = Differential(iv)

    deqs = Equation[]
    new_vars = Term[] # a set of new variables created in each iteration
    transformation = Bijection()

    function create_var()::Num
        new_var_i += 1
        new_var = Symbolics.variable(:y, new_var_i;
                                     T = SymbolicUtils.FnType{Tuple, Real})(iv)
        push!(new_vars, new_var)
        return new_var
    end
    new_var_i = 0

    function change_variables(r::Real)::Real
        return r
    end

    function change_variables(sym::SymbolicUtils.Sym)::SymbolicUtils.Sym
        return sym
    end

    function change_variables(add::SymbolicUtils.Add)::SymbolicUtils.Symbolic
        add_dict = Dict()
        for (term, coeff) in add.dict
            var = change_variables(term)
            add_dict[var] = coeff
        end
        return SymbolicUtils.Add(Real, add.coeff, add_dict)
    end

    function change_variables(mul::SymbolicUtils.Mul)::SymbolicUtils.Symbolic
        mul_dict = Dict()
        for (base, exponent) in mul.dict
            # TODO negative exponent
            if exponent < zero(exponent)
                error("unimplemented")
            end
            var = change_variables(base)
            mul_dict[var] = exponent
        end
        return SymbolicUtils.Mul(Real, mul.coeff, mul_dict)
    end

    function change_variables(div::SymbolicUtils.Div)::SymbolicUtils.Symbolic
        # TODO
        error("unimplemented")
    end

    function change_variables(pow::SymbolicUtils.Pow)::SymbolicUtils.Symbolic
        # TODO negative exponent
        if pow.exp < zero(pow.exp)
            error("unimplemented")
        end
        var = get!(create_var, transformation, pow.base)
        return SymbolicUtils.Pow(var, pow.exp)
    end

    function change_variables(term::SymbolicUtils.Term)::SymbolicUtils.Term
        if term in dvs_set
            return term
        end
        var = get!(create_var, transformation, term)
        args = arguments(term)
        map(change_variables, args)
        op = operation(term)

        if op == sin
            t = SymbolicUtils.Term(cos, args)
            get!(create_var, transformation, t)
        elseif op == cos
            t = SymbolicUtils.Term(sin, args)
            get!(create_var, transformation, t)
        end
        return Symbolics.unwrap(var)
    end

    while true
        polynomial_dicts, residuals = polynomial_coeffs(rhs, dvs)
        # TODO mocking test for semipolynomial

        isz = Symbolics._iszero.(residuals)
        if all(isz)
            append!(deqs, lhs .~ rhs)
            break
        end

        polynomials = map(dict -> SymbolicUtils.Add(Real, 0, dict), polynomial_dicts)

        residuals = convert(Vector{Union{Real, SymbolicUtils.Symbolic}}, residuals)
        for (eq_i, residual) in enumerate(residuals)
            if !isz[eq_i]
                residuals[eq_i] = change_variables(residual)
            end
        end
        rhs = polynomials + residuals
        append!(deqs, lhs .~ rhs)

        g = transformation.(new_vars) # TODO change to a more informative variable name

        # TODO apply `subsitute` on sparse jacobian
        # gₓ = Symbolics.sparsejacobian(g, dvs)
        # @set! gₓ.nzval = substitute.(Symbolics.unwrap.(gₓ.nzval), (transformation,))

        gₓ = Symbolics.jacobian(g, dvs)
        gₓ = substitute.(Symbolics.unwrap.(gₓ), (transformation,))

        rhs = gₓ * Symbolics.rhss(deqs)

        lhs = D.(new_vars)
        append!(dvs, new_vars)
        union!(dvs_set, new_vars)
        empty!(new_vars)
    end

    # TODO observed and defaults
    ODESystem(deqs, iv, dvs, parameters(sys);
              name = Symbol(nameof(sys), "_polynomialized"))
end
