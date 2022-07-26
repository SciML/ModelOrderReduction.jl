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

    function change_variables(sym::SymbolicUtils.Sym)
        throw(ArgumentError("the expression contains the independent variable $sym"))
    end

    function change_variables(add::SymbolicUtils.Add)::SymbolicUtils.Symbolic
        add_dict = Dict()
        sizehint!(add_dict, length(add.dict))
        for (term, coeff) in add.dict
            var = change_variables(term)
            add_dict[var] = coeff
        end
        return SymbolicUtils.Add(Real, add.coeff, add_dict)
    end

    function change_variables(mul::SymbolicUtils.Mul)::SymbolicUtils.Symbolic
        mul_dict = Dict()
        sizehint!(mul_dict, length(mul.dict))
        for (base, exp) in mul.dict
            # use `Pow(base, exp)` as a light version of `Mul(1, Dict(base => exp))`
            pow = SymbolicUtils.Pow(base, exp)
            var = change_variables(pow)
            if var isa SymbolicUtils.Pow
                mul_dict[var.base] = var.exp
            else
                mul_dict[var] = 1
            end
        end
        return SymbolicUtils.Mul(Real, mul.coeff, mul_dict)
    end

    function change_variables(div::SymbolicUtils.Div)::SymbolicUtils.Symbolic
        # TODO
        error("unimplemented")
    end

    function change_variables(pow::SymbolicUtils.Pow)::SymbolicUtils.Symbolic
        base = pow.base
        exp = pow.exp
        if base isa Number
            var = change_variables(exp)
            return SymbolicUtils.Pow(base, var)
        end
        if exp isa Number
            if exp isa Integer
                # TODO negative exponent
                if exp < zero(exp)
                    error("unimplemented")
                end
                var = change_variables(base)
                return SymbolicUtils.Pow(var, exp)
            elseif exp isa Rational
                # TODO
                error("unimplemented")
            end # e.g. AbstractFloat, AbstractIrrational
            throw(ArgumentError(string("polynomialization cannot handle $pow with the ",
                                       "exponent $(typeof(exp)) $exp. Try Integer or ",
                                       "Rational exponent.")))
        end
        throw(ArgumentError(string("polynomialization cannot handle $pow whose base $base ",
                                   "and exponent $exp are both variables.")))
    end

    function change_variables(term::SymbolicUtils.Term)::SymbolicUtils.Symbolic
        if term in dvs_set
            return term
        end
        op = operation(term)
        args = arguments(term)
        if op == (+) || op == (*) || op == (^) || op == (/)
            return mapreduce(change_variables, op, args)
        end
        for arg in args
            change_variables(arg)
        end
        var = get!(create_var, transformation, term)

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
