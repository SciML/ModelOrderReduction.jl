"""
$(TYPEDSIGNATURES)

Compute the DEIM interpolation indices for the given projection basis.

The orthonormal `basis` should not be a sparse matrix. This helper is internal and is
not part of the exported API.
"""
function deim_interpolation_indices(basis::AbstractMatrix)::Vector{Int}
    dim = size(basis, 2)
    indices = Vector{Int}(undef, dim)
    @inbounds @views begin
        r = abs.(basis[:, 1])
        indices[1] = argmax(r)
        for l in 2:dim
            U = basis[:, 1:(l - 1)]
            P = indices[1:(l - 1)]
            PᵀU = U[P, :]
            uₗ = basis[:, l]
            Pᵀuₗ = uₗ[P, :]
            c = vec(PᵀU \ Pᵀuₗ)
            mul!(r, U, c)
            @. r = abs(uₗ - r)
            indices[l] = argmax(r)
        end
    end
    return indices
end

function _sort_observed_equations(equations::Vector{Equation})::Vector{Equation}
    assignments = Dict{Any, Int}()
    for (index, equation) in enumerate(equations)
        assignments[Symbolics.unwrap(equation.lhs)] = index
    end

    dependents = [Int[] for _ in equations]
    degrees = zeros(Int, length(equations))
    for (index, equation) in enumerate(equations)
        dependencies = Set(Symbolics.unwrap.(Symbolics.get_variables(Symbolics.unwrap(equation.rhs))))
        for variable in dependencies
            dependency = get(assignments, variable, nothing)
            if !isnothing(dependency) && dependency != index
                push!(dependents[dependency], index)
                degrees[index] += 1
            end
        end
    end

    available = findall(iszero, degrees)
    ordered = Equation[]
    sizehint!(ordered, length(equations))
    while !isempty(available)
        index = popfirst!(available)
        push!(ordered, equations[index])
        for dependent in dependents[index]
            degrees[dependent] -= 1
            degrees[dependent] == 0 && push!(available, dependent)
        end
    end

    length(ordered) == length(equations) ||
        throw(ArgumentError("observed equations contain a dependency cycle"))
    return ordered
end

"""
$(SIGNATURES)

Compute the reduced model by applying the Discrete Empirical Interpolation Method (DEIM).

This method allows users to input the projection matrices of their choice.

Given the projection matrix ``V\\in\\mathbb R^{n\\times k}`` for the dependent variables
``\\mathbf y\\in\\mathbb R^n`` and the projection matrix
``U\\in\\mathbb R^{n\\times m}`` for the nonlinear function ``\\mathbf F\\in\\mathbb R^n``,
the full-order model (FOM)
```math
\\frac{d}{dt}\\mathbf y(t)=A\\mathbf y(t)+\\mathbf g(t)+\\mathbf F(\\mathbf y(t))
```
is transformed to the reduced-order model (ROM)
```math
\\frac{d}{dt}\\hat{\\mathbf y}(t)=\\underbrace{V^TAV}_{k\\times k}\\hat{\\mathbf y}(t)+V^T
\\mathbf g(t)+\\underbrace{V^TU(P^TU)^{-1}}_{k\\times m}\\underbrace{P^T\\mathbf F(V
\\hat{\\mathbf y}(t))}_{m\\times1}
```
where ``P=[\\mathbf e_{\\rho_1},\\dots,\\mathbf e_{\\rho_m}]\\in\\mathbb R^{n\\times m}``,
``\\rho_1,\\dots,\\rho_m`` are interpolation indices from the DEIM point selection
algorithm, and ``\\mathbf e_{\\rho_i}=[0,\\ldots,0,1,0,\\ldots,0]^T\\in\\mathbb R^n`` is
the ``\\rho_i``-th column of the identity matrix ``I_n\\in\\mathbb R^{n\\times n}``.

# Arguments
- `full_vars::AbstractVector`: the dependent variables
  ``\\underset{n\\times 1}{\\mathbf y}`` in FOM.
- `linear_coeffs::AbstractMatrix`: the coefficient matrix
  ``\\underset{n\\times n}A`` of linear terms in FOM.
- `constant_part::AbstractVector`: the constant terms
  ``\\underset{n\\times 1}{\\mathbf g}`` in FOM.
- `nonlinear_part::AbstractVector`: the nonlinear functions
  ``\\underset{n\\times 1}{\\mathbf F}`` in FOM.
- `reduced_vars::AbstractVector`: the dependent variables
  ``\\underset{k\\times 1}{\\hat{\\mathbf y}}`` in the reduced-order model.
- `linear_projection_matrix::AbstractMatrix`: the projection matrix
  ``\\underset{n\\times k}V`` for the dependent variables ``\\mathbf y``.
- `nonlinear_projection_matrix::AbstractMatrix`: the projection matrix
  ``\\underset{n\\times m}U`` for the nonlinear functions ``\\mathbf F``.

# Keywords
- `kwargs...`: keyword arguments forwarded to `Symbolics.substitute` when constructing
  the nonlinear reduced model.

# Returns
- `Tuple{AbstractVector, Vector{Equation}}`: the reduced right-hand side and
  linear projection equations.
- `linear_projection_eqs`: the linear projection mapping ``\\mathbf y=V\\hat{\\mathbf y}``.

# Throws
- An exception from the matrix operations if the projection matrices have incompatible
  dimensions.

# Examples
```julia
reduced_rhs, projection_equations = deim(
    full_variables, linear_coefficients, constant_terms, nonlinear_terms,
    reduced_variables, state_basis, nonlinear_basis,
)
```
"""
function deim(
        full_vars::AbstractVector, linear_coeffs::AbstractMatrix,
        constant_part::AbstractVector, nonlinear_part::AbstractVector,
        reduced_vars::AbstractVector, linear_projection_matrix::AbstractMatrix,
        nonlinear_projection_matrix::AbstractMatrix; kwargs...
    )
    # rename variables for convenience
    y = full_vars
    A = linear_coeffs
    g = constant_part
    F = nonlinear_part
    ŷ = reduced_vars
    V = linear_projection_matrix
    U = nonlinear_projection_matrix

    # original vars to reduced vars
    linear_projection_eqs = Symbolics.scalarize(y .~ V * ŷ)
    linear_projection_dict = Dict(eq.lhs => eq.rhs for eq in linear_projection_eqs)

    indices = deim_interpolation_indices(U) # DEIM interpolation indices
    # the DEIM projector (not DEIM basis) satisfies
    # F(original_vars) ≈ projector * F(pod_basis * reduced_vars)[indices]
    projector = ((@view U[indices, :])' \ (U' * V))'
    temp = substitute.(F[indices], (linear_projection_dict,); kwargs...)
    F̂ = projector * temp # DEIM approximation for nonlinear func F

    Â = V' * A * V
    ĝ = V' * g
    reduced_rhss = Â * ŷ + ĝ + F̂
    return reduced_rhss, linear_projection_eqs
end

function _is_symbolic_array(expression)
    value = Symbolics.unwrap(expression)
    return value isa AbstractArray || SymbolicUtils.symtype(value) <: AbstractArray
end

function _array_state_groups(unknowns)
    order = Any[]
    rows_by_variable = Dict{Any, Vector{Int}}()
    array_variables = Set{Any}()
    for (row, unknown) in enumerate(unknowns)
        value = Symbolics.unwrap(unknown)
        if SymbolicUtils.iscall(value) && SymbolicUtils.operation(value) === getindex
            variable = first(SymbolicUtils.arguments(value))
            push!(array_variables, variable)
        else
            variable = value
        end
        if !haskey(rows_by_variable, variable)
            rows_by_variable[variable] = Int[]
            push!(order, variable)
        end
        push!(rows_by_variable[variable], row)
    end

    return map(order) do variable
        rows = rows_by_variable[variable]
        expected_rows = collect(first(rows):last(rows))
        rows == expected_rows || throw(
            ArgumentError("the elements of array variable $variable are not contiguous")
        )
        is_array = variable in array_variables
        shape = is_array ? size(Symbolics.wrap(variable)) : ()
        is_array && prod(shape) != length(rows) && throw(
            DimensionMismatch(
                "array variable $variable has shape $shape but $(length(rows)) scalar elements"
            )
        )
        return (; variable, rows, shape, is_array)
    end
end

function _evaluate_nonlinear_snapshot!(
        output::AbstractMatrix, expressions::AbstractVector,
        variables::AbstractVector, snapshot::AbstractMatrix;
        parameter_values::AbstractDict = Dict{Any, Any}(), kwargs...
    )
    replacements = Dict{Any, Any}(parameter_values)
    for column in axes(snapshot, 2)
        for (variable, value) in zip(variables, @view(snapshot[:, column]))
            replacements[variable] = value
            replacements[Num(variable)] = value
        end
        for row in axes(output, 1)
            value = substitute(
                expressions[row], replacements; fold = Val(true), kwargs...
            )
            value = SymbolicUtils.unwrap_const(Symbolics.unwrap(value))
            value isa Number || throw(
                ArgumentError(
                    "nonlinear expression $(expressions[row]) did not evaluate to a number; provide numeric defaults for its parameters"
                )
            )
            output[row, column] = value
        end
    end
    return output
end

function _array_deim(
        source_system::ODESystem, source_snapshot::AbstractMatrix,
        pod_dim::Integer, deim_dim::Integer, name::Symbol; kwargs...
    )
    source_system = deepcopy(source_system)
    source_unknowns = ModelingToolkit.get_unknowns(source_system)
    number_of_source_unknowns = length(source_unknowns)
    size(source_snapshot, 1) == number_of_source_unknowns || throw(
        DimensionMismatch(
            "the snapshot has $(size(source_snapshot, 1)) rows, but the array-form system has $number_of_source_unknowns unknowns"
        )
    )
    state_groups = _array_state_groups(source_unknowns)
    source_observed = ModelingToolkit.get_observed(source_system)

    sys = mtkcompile(source_system)
    @set! sys.name = name
    iv = ModelingToolkit.get_iv(sys)
    D = Differential(iv)
    dynamic_unknowns = ModelingToolkit.get_unknowns(sys)
    model_parameters = filter(ModelingToolkit.get_ps(sys)) do parameter
        !ModelingToolkit.isinitial(parameter)
    end
    model_parameter_keys = Set(
        Symbolics.unwrap(parameter) for parameter in model_parameters
    )
    system_defaults = copy(ModelingToolkit.get_initial_conditions(sys))
    parameter_values = Dict{Any, Any}()
    for parameter in model_parameters
        key = Symbolics.unwrap(parameter)
        haskey(system_defaults, key) || continue
        value = system_defaults[key]
        parameter_values[key] = value
        parameter_values[Symbolics.wrap(key)] = value
    end
    differential_equations, algebraic_equations = get_deqs(sys)
    isempty(algebraic_equations) || throw(
        ArgumentError(
            "the array-form system must structurally compile to an explicit first-order ODE"
        )
    )
    equations_by_unknown = Dict{Any, Equation}()
    for equation in differential_equations
        unknown = only(SymbolicUtils.arguments(Symbolics.unwrap(equation.lhs)))
        haskey(equations_by_unknown, unknown) && throw(
            ArgumentError("the compiled system has duplicate equations for $unknown")
        )
        equations_by_unknown[unknown] = equation
    end
    length(equations_by_unknown) == length(dynamic_unknowns) || throw(
        ArgumentError(
            "the compiled system does not have one differential equation per unknown"
        )
    )
    source_rows = Dict(
        Symbolics.unwrap(unknown) => row for (row, unknown) in enumerate(source_unknowns)
    )
    dynamic_rows = map(dynamic_unknowns) do unknown
        row = get(source_rows, Symbolics.unwrap(unknown), 0)
        iszero(row) && throw(
            ArgumentError(
                "compiled unknown $unknown is not present in the array-form system"
            )
        )
        return row
    end
    dynamic_snapshot = Matrix{Float64}(source_snapshot[dynamic_rows, :])

    state_reducer = POD(dynamic_snapshot, pod_dim)
    reduce!(state_reducer, TSVD())
    state_basis = state_reducer.rbasis

    reduced_name = gensym(:ŷ)
    reduced_state = (@variables $reduced_name(iv)[1:pod_dim])[1]
    @set! sys.unknowns = Symbolics.value.(Symbolics.scalarize(reduced_state))
    ModelingToolkit.get_var_to_name(sys)[
        SymbolicIndexingInterface.getname(reduced_state),
    ] = Symbolics.unwrap(reduced_state)

    right_hand_sides = map(dynamic_unknowns) do unknown
        equation = get(equations_by_unknown, Symbolics.unwrap(unknown), nothing)
        isnothing(equation) && throw(
            ArgumentError("the compiled system has no differential equation for $unknown")
        )
        equation.rhs
    end
    linear_coefficients, constant_part, nonlinear_part = separate_terms(
        right_hand_sides, dynamic_unknowns, iv
    )

    nonlinear_snapshot = similar(dynamic_snapshot)
    _evaluate_nonlinear_snapshot!(
        nonlinear_snapshot, nonlinear_part, dynamic_unknowns, dynamic_snapshot;
        parameter_values, kwargs...
    )
    nonlinear_reducer = POD(nonlinear_snapshot, deim_dim)
    reduce!(nonlinear_reducer, TSVD())
    nonlinear_basis = nonlinear_reducer.rbasis

    reduced_rhs, _ = deim(
        dynamic_unknowns, linear_coefficients, constant_part, nonlinear_part,
        reduced_state, state_basis, nonlinear_basis; kwargs...
    )
    reduced_equation = D(reduced_state) ~ reduced_rhs
    @set! sys.eqs = [reduced_equation]

    reduced_snapshot = state_basis' * dynamic_snapshot
    reconstruction_basis = Matrix{Float64}(source_snapshot) / reduced_snapshot
    reconstruction_basis[dynamic_rows, :] = state_basis
    basis_name = gensym(:reconstruction_basis)
    basis_parameter = (
        @parameters $basis_name[1:number_of_source_unknowns, 1:pod_dim]
    )[1]
    basis_parameter = ModelingToolkit.setdefault(
        basis_parameter, reconstruction_basis
    )
    reduced_scalars = collect(Symbolics.scalarize(reduced_state))
    state_replacements = Dict{Any, Any}()
    reconstruction_equations = map(state_groups) do group
        rows = first(group.rows):last(group.rows)
        reconstruction = Symbolics.wrap(basis_parameter[rows, :] * reduced_scalars)
        value = group.is_array ? reshape(reconstruction, group.shape) :
            only(Symbolics.scalarize(reconstruction))
        state_replacements[group.variable] = value
        Symbolics.wrap(group.variable) ~ value
    end
    source_state_variables = Set(Symbolics.unwrap.(source_unknowns))
    union!(source_state_variables, (group.variable for group in state_groups))
    preserved_observed = Equation[]
    for equation in source_observed
        Symbolics.unwrap(equation.lhs) in source_state_variables && continue
        push!(
            preserved_observed,
            equation.lhs ~ substitute(equation.rhs, state_replacements; kwargs...)
        )
    end
    @set! sys.observed = _sort_observed_equations(
        [
            reconstruction_equations; preserved_observed
        ]
    )
    @set! sys.ps = [model_parameters; Symbolics.unwrap(basis_parameter)]

    initial_conditions = system_defaults
    filter!(pair -> first(pair) in model_parameter_keys, initial_conditions)
    initial_conditions[Symbolics.unwrap(reduced_state)] = reduced_snapshot[:, 1]
    initial_conditions[Symbolics.unwrap(D(reduced_state))] = zeros(pod_dim)
    initial_conditions[Symbolics.unwrap(basis_parameter)] = reconstruction_basis
    @set! sys.initial_conditions = initial_conditions
    @set! sys.initialization_eqs = Equation[]
    guesses = copy(ModelingToolkit.get_guesses(sys))
    empty!(guesses)
    @set! sys.guesses = guesses
    return complete(sys)
end

"""
    $(FUNCTIONNAME)(
        sys::ModelingToolkit.ODESystem,
        snapshot::AbstractMatrix,
        pod_dim::Integer;
        deim_dim::Integer = pod_dim,
        name::Symbol = Symbol(nameof(sys), :_deim),
        kwargs...
    ) -> ModelingToolkit.ODESystem

Reduce a `ModelingToolkit.ODESystem` using the Proper Orthogonal Decomposition (POD) with
the Discrete Empirical Interpolation Method (DEIM).

`snapshot` should be a matrix with the data of each time instance as a column.

The LHS of equations in `sys` are all assumed to be 1st order derivatives. Use
`ModelingToolkit.ode_order_lowering` to transform higher order ODEs before applying DEIM.

`sys` is assumed to have no internal systems. Scalar systems may be passed after
`ModelingToolkit.mtkcompile`. Array-form systems, including the output of MethodOfLines v1
`symbolic_discretize`, should be passed directly. For an array-form system, `snapshot` must
have one row per unknown in that uncompiled system; the returned reduced dynamics remain a
single symbolic array equation and can be used to construct a `DAEProblem` without
scalarizing the reduced system.

The POD basis used for DEIM interpolation is obtained from the snapshot matrix of the
nonlinear terms. For scalar systems this is computed with a runtime-generated function.
For array-form systems it is evaluated symbolically so the offline reduction does not
compile a full-grid output function.

# Arguments
- `sys::ModelingToolkit.ODESystem`: first-order system without internal subsystems. Both
  compiled scalar equations and uncompiled symbolic array equations are supported.
- `snapshot::AbstractMatrix`: state-by-time snapshot matrix for `sys`.
- `pod_dim::Integer`: number of POD state modes to retain.

# Keywords
- `deim_dim::Integer = pod_dim`: number of DEIM modes for nonlinear terms.
- `name::Symbol = Symbol(nameof(sys), :_deim)`: name assigned to the reduced system.
- `kwargs...`: keyword arguments forwarded to ModelingToolkit transformations and
  generated nonlinear functions.

# Examples
```julia
reduced_system = deim(compiled_system, snapshots, 4; deim_dim = 6)
```

# Returns
- `ModelingToolkit.ODESystem`: the reduced and completed system.

# Throws
- `ArgumentError`: if the observed equations contain a dependency cycle.
- `DimensionMismatch`: if an array-form system and its snapshot have different state
  dimensions.
"""
function deim(
        sys::ODESystem, snapshot::AbstractMatrix, pod_dim::Integer;
        deim_dim::Integer = pod_dim, name::Symbol = Symbol(nameof(sys), :_deim),
        kwargs...
    )::ODESystem
    sys = deepcopy(sys)
    uses_array_equations = any(ModelingToolkit.get_eqs(sys)) do eq
        _is_symbolic_array(eq.lhs) || _is_symbolic_array(eq.rhs)
    end
    uses_array_equations && return _array_deim(
        sys, snapshot, pod_dim, deim_dim, name; kwargs...
    )
    @set! sys.name = name

    # handle ODESystem.substitutions
    # https://github.com/SciML/ModelingToolkit.jl/issues/1754
    sys = tearing_substitution(sys; kwargs...)

    iv = ModelingToolkit.get_iv(sys) # the single independent variable
    D = Differential(iv)
    dvs = ModelingToolkit.get_unknowns(sys) # dependent variables

    pod_reducer = POD(snapshot, pod_dim)
    reduce!(pod_reducer, TSVD())
    V = pod_reducer.rbasis # POD basis

    var_name = gensym(:ŷ)
    ŷ = (@variables $var_name(iv)[1:pod_dim])[1]
    @set! sys.unknowns = Symbolics.value.(Symbolics.scalarize(ŷ)) # new variables from POD
    ModelingToolkit.get_var_to_name(sys)[SymbolicIndexingInterface.getname(ŷ)] = Symbolics.unwrap(ŷ)

    deqs, eqs = get_deqs(sys) # split eqs into differential and non-differential equations
    rhs = [eq.rhs for eq in deqs]
    # a sparse matrix of coefficients for the linear part,
    # a vector of constant terms and a vector of nonlinear terms about dvs
    A, g, F = separate_terms(rhs, dvs, iv)

    # generate an in-place function from the symbolic expression of the nonlinear functions
    F_func! = build_function(F, dvs; expression = Val{false}, kwargs...)[2]
    nonlinear_snapshot = similar(snapshot) # snapshot matrix of nonlinear terms
    for i in 1:size(snapshot, 2) # iterate through time instances
        F_func!(view(nonlinear_snapshot, :, i), view(snapshot, :, i))
    end

    deim_reducer = POD(nonlinear_snapshot, deim_dim)
    reduce!(deim_reducer, TSVD())
    U = deim_reducer.rbasis # DEIM projection basis

    reduced_rhss, linear_projection_eqs = deim(dvs, A, g, F, ŷ, V, U; kwargs...)

    reduced_deqs = D.(ŷ) ~ reduced_rhss
    @set! sys.eqs = [Symbolics.scalarize(reduced_deqs); eqs]

    old_observed = ModelingToolkit.get_observed(sys)
    new_observed = [old_observed; linear_projection_eqs]
    @set! sys.observed = _sort_observed_equations(new_observed)

    # Numeric initial conditions for the reduced unknowns from the snapshot's first column.
    # The snapshot is assumed to start at t = tspan[1], matching the FOM initial state.
    new_ics = copy(ModelingToolkit.get_initial_conditions(sys))
    new_ics[Symbolics.unwrap(ŷ)] = V' * snapshot[:, 1]
    @set! sys.initial_conditions = new_ics

    return complete(sys)
end
