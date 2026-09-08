"""
$(TYPEDSIGNATURES)

Return the array (or scalar) unknown differentiated on the left-hand side of a
differential equation.
"""
function _differential_unknown(equation::Equation)
    lhs = Symbolics.unwrap(equation.lhs)
    return only(SymbolicUtils.arguments(lhs))
end

"""
$(TYPEDSIGNATURES)

Whether `expression` is a symbolic array.
"""
function _is_symbolic_array(expression)
    return SymbolicUtils.symtype(Symbolics.unwrap(expression)) <: AbstractArray
end

"""
$(TYPEDSIGNATURES)

Return true when the source system has no algebraic equations and every differential
unknown is covered by an equation whose left-hand side differentiates a source field
variable. In that case Galerkin projection can substitute array fields directly and keep
an ``O(1)`` symbolic residual.
"""
function _can_array_galerkin(fom::FullOrderModel)
    differential, algebraic = _split_equations(ModelingToolkit.equations(fom.system))
    isempty(algebraic) || return false
    length(differential) == length(fom.fields) || return false
    # Array-field residuals stay as broadcast/array ops; pure scalar sources use the
    # compiled Galerkin path, which also projects numeric linear terms exactly.
    any(eq -> _is_symbolic_array(_differential_unknown(eq)), differential) || return false

    field_variables = Set{Any}(field.variable for field in fom.fields)
    covered = Set{Any}()
    for equation in differential
        unknown = _differential_unknown(equation)
        unknown in field_variables || return false
        unknown in covered && return false
        push!(covered, unknown)
    end
    return covered == field_variables
end

"""
$(TYPEDSIGNATURES)

Build the Galerkin right-hand side ``V^T f(V\\hat y)`` from the source system's field
equations as array linear algebra. Each field's state-basis block and its transpose are
array parameters, so the generated residual graph does not grow with the full-order
dimension.
"""
function _galerkin_array_rhs(
        fom::FullOrderModel, V::AbstractMatrix, reduced_state, taken::Set{Symbol};
        kwargs...
    )
    parameters = Any[]
    values = Dict{Any, Any}()
    function array_parameter(base, value)
        parameter, pair = _array_parameter(base, value, taken)
        push!(parameters, first(pair))
        push!(values, pair)
        return parameter
    end

    dynamic_pos = Dict{Int, Int}(row => i for (i, row) in enumerate(fom.rows))
    field_positions = Dict{Any, Vector{Int}}()
    replacements = Dict{Any, Any}()
    position_to_lift = Dict{Int, Any}()
    for field in fom.fields
        positions = Int[]
        for row in field.rows
            position = get(dynamic_pos, row, 0)
            iszero(position) && return nothing
            push!(positions, position)
        end
        field_positions[field.variable] = positions
        V_field = array_parameter(:V, Matrix{Float64}(V[positions, :]))
        lift = Symbolics.wrap(V_field * reduced_state)
        value = isempty(field.shape) ? only(Symbolics.scalarize(lift)) :
            reshape(lift, field.shape)
        replacements[field.variable] = value
        for (j, row) in enumerate(field.rows)
            element = lift[j]
            replacements[fom.source_unknowns[row]] = element
            position_to_lift[positions[j]] = element
        end
    end
    for (i, unknown) in enumerate(fom.unknowns)
        replacements[unknown] = position_to_lift[i]
    end

    differential, _ = _split_equations(ModelingToolkit.equations(fom.system))
    rhs = nothing
    for equation in differential
        unknown = _differential_unknown(equation)
        positions = field_positions[unknown]
        Vt_field = array_parameter(:Vt, Matrix{Float64}(V[positions, :]'))
        residual = substitute(equation.rhs, replacements; kwargs...)
        if !_is_symbolic_array(residual) && length(positions) == 1
            residual = Symbolics.wrap([residual])
        end
        contrib = Vt_field * residual
        rhs = isnothing(rhs) ? contrib : rhs + contrib
    end
    return rhs, parameters, values
end

"""
$(TYPEDSIGNATURES)

Numeric Galerkin residual ``V^T f(V\\hat y)`` for the source field equations at the
reduced state `reduced_state` and time `time` (`nothing` for autonomous models).
"""
function _galerkin_array_derivative(
        fom::FullOrderModel, V::AbstractMatrix, reduced_state::AbstractVector, time;
        kwargs...
    )
    lifted = reshape(V * reduced_state, :, 1)
    dynamic_pos = Dict{Int, Int}(row => i for (i, row) in enumerate(fom.rows))
    replacements = Dict{Any, Any}(fom.parameter_values)
    isnothing(time) || (replacements[fom.iv] = time)
    for (i, unknown) in enumerate(fom.unknowns)
        replacements[unknown] = lifted[i]
    end
    for field in fom.fields
        positions = [dynamic_pos[row] for row in field.rows]
        values = lifted[positions]
        replacements[field.variable] = isempty(field.shape) ? only(values) :
            reshape(values, field.shape)
        for (j, row) in enumerate(field.rows)
            replacements[fom.source_unknowns[row]] = values[j]
        end
    end

    field_by_variable = Dict{Any, SourceField}(
        field.variable => field for field in fom.fields
    )
    derivative = zeros(size(V, 2))
    differential, _ = _split_equations(ModelingToolkit.equations(fom.system))
    for equation in differential
        unknown = _differential_unknown(equation)
        positions = [dynamic_pos[row] for row in field_by_variable[unknown].rows]
        residual = substitute(equation.rhs, replacements; fold = Val(true), kwargs...)
        residual = SymbolicUtils.unwrap_const(Symbolics.unwrap(residual))
        if residual isa Number
            residual = [residual]
        elseif residual isa AbstractArray
            residual = vec(Array(residual))
        else
            throw(
                ArgumentError(
                    "expression $(equation.rhs) did not evaluate to a number; provide numeric parameter values and snapshot_times for time-dependent expressions"
                )
            )
        end
        derivative += V[positions, :]' * residual
    end
    return derivative
end

"""
$(TYPEDEF)

Numeric POD Galerkin projection of a [`FullOrderModel`](@ref) when the source equations are
scalarized.

The full-order model
```math
\\frac{d}{dt}\\mathbf y(t)=A\\mathbf y(t)+\\mathbf g(t)+\\mathbf F(\\mathbf y(t),t)
```
is projected onto the state basis ``V`` as
```math
\\frac{d}{dt}\\hat{\\mathbf y}(t)=V^TAV\\hat{\\mathbf y}(t)+V^T\\mathbf g(t)+V^T
\\mathbf F(V\\hat{\\mathbf y}(t),t).
```
Unlike [`Projection`](@ref), every nonlinear residual row is retained, so the online cost
scales with the full-order dimension.
"""
struct GalerkinProjection
    "state basis ``V``"
    state_basis::Matrix{Float64}
    "projected linear coefficients ``V^TAV``"
    linear::Matrix{Float64}
    "projection ``V^T\\mathbf g`` of the numeric forcing entries"
    forcing::Vector{Float64}
    "distinct symbolic forcing expressions"
    forcing_terms::Vector{Num}
    "column ``j`` sums the rows of ``V`` carrying `forcing_terms[j]`"
    forcing_map::Matrix{Float64}
    "rows of ``\\mathbf F`` that are not identically zero"
    indices::Vector{Int}
    "dynamic unknowns referenced by the retained nonlinear terms"
    stencil::Vector{Int}
    "Galerkin projector ``V^T`` restricted to `indices`"
    projector::Matrix{Float64}
end

function GalerkinProjection(fom::FullOrderModel, V::AbstractMatrix)
    indices = findall(!iszero, fom.nonlinear)
    if isempty(indices)
        stencil = Int[]
        projector = zeros(size(V, 2), 0)
    else
        # Lift the full dynamic state so arbitrary residual coupling stays valid, and
        # project only the retained residual rows with the matching columns of ``V^T``.
        stencil = collect(eachindex(fom.unknowns))
        projector = Matrix{Float64}(V[indices, :]')
    end

    numeric_forcing = zeros(length(fom.forcing))
    forcing_terms = Num[]
    forcing_map = zeros(size(V, 2), 0)
    term_columns = Dict{Num, Int}()
    for (row, term) in enumerate(fom.forcing)
        value = SymbolicUtils.unwrap_const(Symbolics.unwrap(term))
        if value isa Number
            numeric_forcing[row] = value
            continue
        end
        column = get(term_columns, term, 0)
        if iszero(column)
            push!(forcing_terms, term)
            forcing_map = [forcing_map zeros(size(V, 2))]
            column = term_columns[term] = length(forcing_terms)
        end
        forcing_map[:, column] += V[row, :]
    end
    return GalerkinProjection(
        Matrix{Float64}(V), Matrix{Float64}(V' * fom.linear * V), V' * numeric_forcing,
        forcing_terms, forcing_map, indices, stencil, projector
    )
end

"""
$(TYPEDSIGNATURES)

Build the symbolic Galerkin right-hand side for a scalarized [`FullOrderModel`](@ref) as
array linear algebra in the reduced state.
"""
function _galerkin_compiled_rhs(
        fom::FullOrderModel, projection::GalerkinProjection, reduced_state,
        taken::Set{Symbol}; kwargs...
    )
    parameters = Any[]
    values = Dict{Any, Any}()
    function array_parameter(base, value)
        parameter, pair = _array_parameter(base, value, taken)
        push!(parameters, first(pair))
        push!(values, pair)
        return parameter
    end

    rhs = array_parameter(:A, projection.linear) * reduced_state
    if any(!iszero, projection.forcing)
        rhs += array_parameter(:g, projection.forcing)
    end
    if !isempty(projection.forcing_terms)
        rhs += array_parameter(:G, projection.forcing_map) *
            Symbolics.wrap(projection.forcing_terms)
    end
    if !isempty(projection.indices)
        lifted = array_parameter(:V, projection.state_basis[projection.stencil, :]) *
            reduced_state
        replacements = Dict{Any, Any}(
            fom.unknowns[j] => lifted[i] for (i, j) in enumerate(projection.stencil)
        )
        sampled = [
            substitute(fom.nonlinear[i], replacements; kwargs...) for i in projection.indices
        ]
        rhs += array_parameter(:Vt, projection.projector) * Symbolics.wrap(sampled)
    end
    return rhs, parameters, values
end

"""
$(TYPEDSIGNATURES)

Evaluate the Galerkin right-hand side of a scalarized projection at `reduced_state`.
"""
function _galerkin_compiled_derivative(
        fom::FullOrderModel, projection::GalerkinProjection, reduced_state::AbstractVector,
        time; kwargs...
    )
    lifted = reshape(projection.state_basis * reduced_state, :, 1)
    times = isnothing(time) ? nothing : [time]
    derivative = projection.linear * reduced_state + projection.forcing
    if !isempty(projection.forcing_terms)
        forcing = _evaluate(
            projection.forcing_terms, fom.unknowns, lifted, fom.parameter_values, fom.iv,
            times; kwargs...
        )
        derivative += projection.forcing_map * vec(forcing)
    end
    if !isempty(projection.indices)
        sampled = _evaluate(
            fom.nonlinear[projection.indices], fom.unknowns, lifted, fom.parameter_values,
            fom.iv, times; kwargs...
        )
        derivative += projection.projector * vec(sampled)
    end
    return derivative
end

function _pod(
        sys::System, snapshot::AbstractMatrix, pod_dim::Integer, name::Symbol;
        snapshot_times = nothing, training_parameters = Dict{Any, Any}(),
        tspan = nothing, kwargs...
    )
    rows = length(ModelingToolkit.unknowns(sys))
    size(snapshot, 1) == rows || throw(
        DimensionMismatch(
            "the snapshot has $(size(snapshot, 1)) rows, but the source system has $rows unknowns"
        )
    )
    isnothing(snapshot_times) || length(snapshot_times) == size(snapshot, 2) ||
        throw(DimensionMismatch("snapshot_times must contain one time per snapshot column"))

    fom = FullOrderModel(sys; training_parameters)
    states = Matrix{Float64}(snapshot[fom.rows, :])
    state_basis = _pod_basis(states, pod_dim)

    dim = size(state_basis, 2)
    taken = _reserved_names(fom)
    state_name = _generated_name(:ŷ, taken)
    reduced_state = (@variables $state_name(fom.iv)[1:dim])[1]
    time0 = isnothing(snapshot_times) ? nothing : first(snapshot_times)

    if _can_array_galerkin(fom)
        result = _galerkin_array_rhs(
            fom, state_basis, reduced_state, taken; kwargs...
        )
        if !isnothing(result)
            rhs, array_parameters, array_values = result
            initial_derivative = _galerkin_array_derivative(
                fom, state_basis, state_basis' * states[:, 1], time0; kwargs...
            )
            return _assemble_reduced_system(
                fom, snapshot, state_basis, reduced_state, rhs, array_parameters,
                array_values, taken, name, initial_derivative; tspan, kwargs...
            )
        end
    end

    projection = GalerkinProjection(fom, state_basis)
    rhs, array_parameters, array_values = _galerkin_compiled_rhs(
        fom, projection, reduced_state, taken; kwargs...
    )
    initial_derivative = _galerkin_compiled_derivative(
        fom, projection, state_basis' * states[:, 1], time0; kwargs...
    )
    return _assemble_reduced_system(
        fom, snapshot, state_basis, reduced_state, rhs, array_parameters, array_values,
        taken, name, initial_derivative; tspan, kwargs...
    )
end

"""
    $(FUNCTIONNAME)(
        sys::ModelingToolkit.System,
        snapshot::AbstractMatrix,
        pod_dim::Integer;
        name::Symbol = Symbol(nameof(sys), :_pod),
        snapshot_times = nothing,
        kwargs...
    ) -> ModelingToolkit.System

Reduce a first-order `ModelingToolkit.System` with Proper Orthogonal Decomposition (POD)
Galerkin projection, without Discrete Empirical Interpolation (DEIM).

The rows of `snapshot` follow `ModelingToolkit.unknowns(sys)` and each column is one time
instance. The state basis is the POD basis of `snapshot`, computed with [`TSVD`](@ref).

When the source system is an explicit first-order model whose differential equations are
written in terms of the source field variables (including array equations), the reduced
dynamics are assembled by substituting ``\\mathbf y = V\\hat{\\mathbf y}`` into those
equations and left-multiplying by ``V^T``. The state basis and its transpose are array
parameters, so the generated residual is array linear algebra whose symbolic size does not
grow with the full-order grid. When the source has already been scalarized, the same
Galerkin projection is applied to the compiled residual; the online nonlinear cost then
scales with the full-order dimension (use [`deim`](@ref) to hyper-reduce it).

`sys` may contain algebraic equations or array equations. It is structurally simplified as
needed. The returned system always has one array differential equation for the reduced
state and one reconstruction observed equation per source field. Unknowns eliminated during
simplification are reconstructed by a least-squares fit to `snapshot`.

The reduced system inherits the time span of `sys`, so a problem can be built from it
without repeating one. Passing a time span explicitly still overrides it, and a source
system without one produces a reduced system without one.

Construct a `DAEProblem` with `build_initializeprob = false` from the returned system to
keep array code generation, or call `ModelingToolkit.mtkcompile` on it and construct an
`ODEProblem` to generate scalar code for the reduced equation. Reconstruct fields with
`SymbolicIndexingInterface.observed(reduced_system, field)`.

# Arguments
- `sys::ModelingToolkit.System`: first-order source system without subsystems.
- `snapshot::AbstractMatrix`: state-by-time snapshot matrix for `sys`.
- `pod_dim::Integer`: number of POD state modes to retain.

# Keywords
- `name::Symbol = Symbol(nameof(sys), :_pod)`: name of the reduced system.
- `snapshot_times = nothing`: time of each snapshot column; required when the equations
  depend on the independent variable.
- `kwargs...`: keyword arguments forwarded to `Symbolics.substitute`.

# Returns
- `ModelingToolkit.System`: the completed reduced system.

# Throws
- `DimensionMismatch`: if `snapshot` does not have one row per unknown of `sys`, or if
  `snapshot_times` does not have one entry per column.
- `ArgumentError`: if `sys` does not simplify to an explicit first-order ODE, or if a
  residual term cannot be evaluated numerically.

# Examples
```julia
reduced_system = pod(source_system, snapshots, 4)
```
"""
function pod(
        sys::System, snapshot::AbstractMatrix, pod_dim::Integer;
        name::Symbol = Symbol(nameof(sys), :_pod), snapshot_times = nothing, kwargs...
    )::System
    return _pod(
        sys, snapshot, pod_dim, name;
        snapshot_times, tspan = ModelingToolkit.get_tspan(sys), kwargs...
    )
end

"""
    $(FUNCTIONNAME)(
        prob::Union{SciMLBase.AbstractODEProblem, SciMLBase.AbstractDAEProblem},
        sol,
        pod_dim::Integer;
        name::Union{Nothing, Symbol} = nothing,
        kwargs...
    ) -> ModelingToolkit.System

Reduce the symbolic system behind `prob` with POD Galerkin projection, using one of its
saved solutions `sol` as the training snapshot.

`prob` must have been constructed from a `ModelingToolkit.System`, such as the array-form
`DAEProblem` returned by MethodOfLines. The saved states are the snapshot columns, the
saved times supply the independent variable for time-dependent terms, and the parameter
values of `prob` are used for training and become the defaults of the reduced system.
The reduced system also inherits the time span of `prob`, which is the interval it was
trained on, so a problem can be built from it without repeating one.
`sol` may be the `SciMLBase.PDETimeSeriesSolution` returned by MethodOfLines or its
underlying `SciMLBase.AbstractODESolution`. See the system method for the reduction itself
and for how to construct problems from the returned system.

# Arguments
- `prob`: symbolic first-order problem whose function stores a `ModelingToolkit.System`.
- `sol`: saved solution obtained from `prob`.
- `pod_dim::Integer`: number of POD state modes to retain.

# Keywords
- `name::Union{Nothing, Symbol} = nothing`: name of the reduced system. The default appends
  `_pod` to the name of the full system.
- `kwargs...`: keyword arguments forwarded to `Symbolics.substitute`.

# Returns
- `ModelingToolkit.System`: the completed reduced system.

# Throws
- `ArgumentError`: if `prob` has no symbolic system, if `sol` was not obtained from `prob`
  or does not start at the beginning of `prob`, or if the system does not simplify to an
  explicit first-order ODE.
- `DimensionMismatch`: if the saved states do not match the unknowns of the system.

# Examples
```julia
full_problem = discretize(pde_system, discretization; fallback = false)
full_solution = solve(full_problem)
reduced_system = pod(full_problem, full_solution, 4)
reduced_problem = DAEProblem(
    reduced_system, nothing; build_initializeprob = false
)
```
"""
function pod(
        prob::SymbolicProblem, sol::SciMLBase.AbstractODESolution, pod_dim::Integer;
        name::Union{Nothing, Symbol} = nothing, kwargs...
    )::System
    sys = SymbolicIndexingInterface.symbolic_container(prob.f)
    sys isa System ||
        throw(ArgumentError("the problem must contain a symbolic ModelingToolkit system"))
    SymbolicIndexingInterface.symbolic_container(sol.prob.f) === sys ||
        throw(ArgumentError("the solution must have been obtained from the supplied problem"))
    first(sol.t) == first(prob.tspan) ||
        throw(ArgumentError("the solution must save the state at the start of the problem"))
    snapshot = Array(sol)
    size(snapshot, 1) == length(ModelingToolkit.unknowns(sys)) ||
        throw(DimensionMismatch("solution states must match the system unknowns"))

    training_parameters = Dict{Any, Any}()
    for parameter in ModelingToolkit.parameters(sys)
        ModelingToolkit.isinitial(parameter) && continue
        training_parameters[Symbolics.unwrap(parameter)] = prob.ps[parameter]
    end
    reduced_name = isnothing(name) ? Symbol(nameof(sys), :_pod) : name
    return _pod(
        sys, snapshot, pod_dim, reduced_name;
        snapshot_times = sol.t, training_parameters, tspan = prob.tspan, kwargs...
    )
end

function pod(
        prob::SymbolicProblem, sol::SciMLBase.PDETimeSeriesSolution, pod_dim::Integer;
        kwargs...
    )::System
    return pod(prob, sol.original_sol, pod_dim; kwargs...)
end
