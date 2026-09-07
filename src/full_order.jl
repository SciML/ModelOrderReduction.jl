"""
    $(TYPEDEF)

Scalar unknowns of a source system that belong to one symbolic variable.

# Fields
$(TYPEDFIELDS)
"""
struct SourceField
    "the array variable, or the scalar unknown itself"
    variable::Any
    "snapshot rows of the elements, in column-major order for an array variable"
    rows::Vector{Int}
    "array shape, or `()` for a scalar unknown"
    shape::Tuple{Vararg{Int}}
end

"""
    $(TYPEDEF)

Explicit first-order form of a source system, prepared for projection.

Structural simplification eliminates the algebraic equations of the source system. The
remaining differential unknowns are the dynamic unknowns, and their right-hand sides are
split as `linear * unknowns + forcing + nonlinear`: `linear` holds the numeric linear
coefficients, `forcing` holds the terms free of dynamic unknowns, and `nonlinear` holds
every other state-dependent term, including linear terms with symbolic coefficients.

This type is internal and is not part of the public API.

# Fields
$(TYPEDFIELDS)
"""
struct FullOrderModel{S, T}
    "source system, before structural simplification"
    system::S
    "independent variable"
    iv::T
    "scalar unknowns of the source system, in snapshot row order"
    source_unknowns::Vector{Any}
    "source unknowns grouped by symbolic variable"
    fields::Vector{SourceField}
    "dynamic unknowns of the explicit first-order form"
    unknowns::Vector{Any}
    "snapshot row of each dynamic unknown"
    rows::Vector{Int}
    "numeric linear coefficients of the dynamic right-hand sides"
    linear::SparseMatrixCSC{Float64, Int}
    "state-independent terms of the dynamic right-hand sides"
    forcing::Vector{Num}
    "state-dependent terms of the dynamic right-hand sides handled by interpolation"
    nonlinear::Vector{Num}
    "model parameters of the explicit form"
    parameters::Vector{Any}
    "numeric parameter values used for training and as reduced-model defaults"
    parameter_values::Dict{Any, Any}
    "source observed equations that do not define a source unknown"
    observed::Vector{Equation}
end

function FullOrderModel(source::System; training_parameters = Dict{Any, Any}())
    source_unknowns = Any[Symbolics.unwrap(unknown) for unknown in ModelingToolkit.unknowns(source)]
    fields = _source_fields(source_unknowns)
    sys = _explicit_ode(source)
    iv = ModelingToolkit.independent_variable(sys)
    unknowns = Any[Symbolics.unwrap(unknown) for unknown in ModelingToolkit.unknowns(sys)]

    differential, algebraic = _split_equations(ModelingToolkit.full_equations(sys))
    isempty(algebraic) || throw(
        ArgumentError("the source system must structurally compile to an explicit first-order ODE")
    )
    right_hand_sides = Dict{Any, Any}()
    for equation in differential
        unknown = only(SymbolicUtils.arguments(Symbolics.unwrap(equation.lhs)))
        haskey(right_hand_sides, unknown) && throw(
            ArgumentError("the compiled system has duplicate equations for $unknown")
        )
        right_hand_sides[unknown] = equation.rhs
    end
    length(right_hand_sides) == length(unknowns) || throw(
        ArgumentError("the compiled system does not have one differential equation per unknown")
    )
    rhs = map(unknowns) do unknown
        get(right_hand_sides, unknown) do
            throw(ArgumentError("the compiled system has no differential equation for $unknown"))
        end
    end
    source_rows = Dict{Any, Int}(unknown => row for (row, unknown) in enumerate(source_unknowns))
    rows = map(unknowns) do unknown
        get(source_rows, unknown) do
            throw(ArgumentError("compiled unknown $unknown is not present in the source system"))
        end
    end
    linear, forcing, nonlinear = separate_terms(rhs, unknowns)

    parameters = Any[
        Symbolics.unwrap(parameter) for parameter in ModelingToolkit.parameters(sys)
            if !ModelingToolkit.isinitial(parameter)
    ]
    defaults = ModelingToolkit.initial_conditions(sys)
    parameter_values = Dict{Any, Any}()
    for parameter in parameters
        if haskey(training_parameters, parameter)
            value = training_parameters[parameter]
        elseif haskey(defaults, parameter)
            value = defaults[parameter]
        else
            continue
        end
        parameter_values[parameter] = SymbolicUtils.unwrap_const(Symbolics.unwrap(value))
    end

    state_variables = Set{Any}(source_unknowns)
    union!(state_variables, (field.variable for field in fields))
    observed = filter(ModelingToolkit.observed(source)) do equation
        !(Symbolics.unwrap(equation.lhs) in state_variables)
    end
    return FullOrderModel(
        source, iv, source_unknowns, fields, unknowns, rows, linear, forcing, nonlinear,
        parameters, parameter_values, observed
    )
end

_is_differential(equation::Equation) = (
    lhs = Symbolics.unwrap(equation.lhs);
    SymbolicUtils.iscall(lhs) && SymbolicUtils.operation(lhs) isa Differential
)

function _split_equations(equations)
    differential = filter(_is_differential, equations)
    algebraic = filter(!_is_differential, equations)
    return differential, algebraic
end

function _explicit_ode(source::System)
    differential, algebraic = _split_equations(ModelingToolkit.equations(source))
    if isempty(algebraic) && length(differential) == length(ModelingToolkit.unknowns(source))
        return source
    end
    return mtkcompile(source)
end

function _source_fields(unknowns)
    order = Any[]
    elements = Dict{Any, Vector{Tuple{Int, Tuple}}}()
    for (row, unknown) in enumerate(unknowns)
        if SymbolicUtils.iscall(unknown) && SymbolicUtils.operation(unknown) === getindex
            index_arguments = SymbolicUtils.arguments(unknown)
            variable = first(index_arguments)
            indices = Tuple(SymbolicUtils.unwrap_const.(index_arguments[2:end]))
        else
            variable = unknown
            indices = ()
        end
        haskey(elements, variable) || (elements[variable] = []; push!(order, variable))
        push!(elements[variable], (row, indices))
    end
    return map(order) do variable
        entries = elements[variable]
        isempty(last(first(entries))) && return SourceField(variable, [first(only(entries))], ())
        shape = size(Symbolics.wrap(variable))
        length(entries) == prod(shape) || throw(
            DimensionMismatch(
                "array variable $variable has shape $shape but $(length(entries)) scalar elements"
            )
        )
        rows = zeros(Int, prod(shape))
        linear_indices = LinearIndices(shape)
        for (row, indices) in entries
            all(index -> index isa Integer, indices) || throw(
                ArgumentError("array variable $variable has non-integer indices $indices")
            )
            position = linear_indices[indices...]
            iszero(rows[position]) || throw(
                ArgumentError("array variable $variable repeats index $indices")
            )
            rows[position] = row
        end
        return SourceField(variable, rows, shape)
    end
end

"""
$(TYPEDSIGNATURES)

Split each expression in `exprs` as `linear * vars + forcing + nonlinear`.

`linear` is a sparse matrix holding the numeric coefficients of terms that are linear in
`vars`. `forcing` collects the additive terms that do not depend on `vars`. `nonlinear`
collects every remaining term, so it also holds terms that are linear in `vars` with a
symbolic coefficient.

Variables in `vars` must be unique.
"""
function separate_terms(exprs::AbstractVector, vars::AbstractVector)
    vars = Symbolics.unwrap.(vars)
    length(Set(vars)) == length(vars) || throw(ArgumentError("vars: $vars are not unique"))
    coefficients, residual = Symbolics.semilinear_form(exprs, vars)

    forcing = fill(Num(0), length(exprs))
    nonlinear = fill(Num(0), length(exprs))
    linear_I = Int[]
    linear_J = Int[]
    linear_V = Float64[]
    for (i, j, coefficient) in zip(findnz(coefficients)...)
        value = SymbolicUtils.unwrap_const(Symbolics.unwrap(coefficient))
        if value isa Number
            push!(linear_I, i)
            push!(linear_J, j)
            push!(linear_V, value)
        else
            nonlinear[i] += coefficient * vars[j]
        end
    end
    linear = sparse(linear_I, linear_J, linear_V, length(exprs), length(vars))

    state = Set{Any}(vars)
    for (i, expr) in enumerate(residual)
        for term in _additive_terms(expr)
            if any(in(state), Symbolics.unwrap.(Symbolics.get_variables(term)))
                nonlinear[i] += term
            else
                forcing[i] += term
            end
        end
    end
    return linear, forcing, nonlinear
end

function _additive_terms(expr)
    value = Symbolics.unwrap(expr)
    if SymbolicUtils.iscall(value) && SymbolicUtils.operation(value) === (+)
        return SymbolicUtils.arguments(value)
    end
    return (value,)
end

"""
$(TYPEDSIGNATURES)

Evaluate `expressions` at every column of `snapshot`, whose rows follow `variables`.

`parameter_values` supplies numeric parameter values and `times` the value of the
independent variable `iv` for each column. Returns a matrix with one row per expression.
"""
function _evaluate(
        expressions::AbstractVector, variables::AbstractVector, snapshot::AbstractMatrix,
        parameter_values::AbstractDict, iv, times; kwargs...
    )
    output = zeros(Float64, length(expressions), size(snapshot, 2))
    nonzero_rows = findall(!iszero, expressions)
    replacements = Dict{Any, Any}(parameter_values)
    for column in axes(snapshot, 2)
        isnothing(times) || (replacements[iv] = times[column])
        for (variable, value) in zip(variables, @view(snapshot[:, column]))
            replacements[variable] = value
        end
        for row in nonzero_rows
            value = substitute(expressions[row], replacements; fold = Val(true), kwargs...)
            value = SymbolicUtils.unwrap_const(Symbolics.unwrap(value))
            value isa Number || throw(
                ArgumentError(
                    "expression $(expressions[row]) did not evaluate to a number; provide numeric parameter values and snapshot_times for time-dependent expressions"
                )
            )
            output[row, column] = value
        end
    end
    return output
end
