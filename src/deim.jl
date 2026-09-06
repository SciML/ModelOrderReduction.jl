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

function _pod_basis(snapshot::AbstractMatrix, dim::Integer)
    reducer = POD(snapshot, dim)
    reduce!(reducer, TSVD())
    return reducer.rbasis
end

"""
$(TYPEDSIGNATURES)

Galerkin-project the right-hand side of `fom` onto the state basis `V`, approximating the
nonlinear part by DEIM in the basis `U`.

With the DEIM interpolation indices ``\\rho_1,\\dots,\\rho_m`` of ``U`` and the
selection matrix ``P=[\\mathbf e_{\\rho_1},\\dots,\\mathbf e_{\\rho_m}]``, the full-order
model
```math
\\frac{d}{dt}\\mathbf y(t)=A\\mathbf y(t)+\\mathbf g(t)+\\mathbf F(\\mathbf y(t),t)
```
becomes
```math
\\frac{d}{dt}\\hat{\\mathbf y}(t)=\\underbrace{V^TAV}_{k\\times k}\\hat{\\mathbf y}(t)+V^T
\\mathbf g(t)+\\underbrace{V^TU(P^TU)^{-1}}_{k\\times m}\\underbrace{P^T\\mathbf F(V
\\hat{\\mathbf y}(t),t)}_{m\\times1}.
```
Only the ``m`` sampled entries of ``\\mathbf F`` are lifted to the reduced state
`reduced_state`, so the returned array expression does not grow with the full-order
dimension. `kwargs` are forwarded to `Symbolics.substitute`.
"""
function _reduced_rhs(
        fom::FullOrderModel, reduced_state, V::AbstractMatrix, U::AbstractMatrix; kwargs...
    )
    indices = deim_interpolation_indices(U)
    sampled = fom.nonlinear[indices]
    column = Dict{Any, Int}(unknown => j for (j, unknown) in enumerate(fom.unknowns))
    lift = Dict{Any, Any}()
    for expression in sampled, variable in Symbolics.get_variables(expression)
        variable = Symbolics.unwrap(variable)
        j = get(column, variable, 0)
        iszero(j) && continue
        lift[variable] = sum(V[j, l] * reduced_state[l] for l in axes(V, 2))
    end
    sampled = [substitute(expression, lift; kwargs...) for expression in sampled]
    projector = ((@view U[indices, :])' \ (U' * V))'

    rhs = (V' * fom.linear * V) * reduced_state
    forcing = V' * fom.forcing
    all(iszero, forcing) || (rhs += forcing)
    return Symbolics.wrap(rhs + projector * sampled)
end

"""
$(TYPEDSIGNATURES)

Assemble the reduced system for `fom` from the state basis `V`, the DEIM basis `U`, and
the training `snapshot` whose rows follow the source unknowns.

The system has one array differential equation for the reduced state, one array (or
scalar) reconstruction observed equation per source field, and the source observed
equations rewritten in terms of the reduced state. Dynamic source unknowns are
reconstructed with `V`; unknowns eliminated by structural simplification use a
least-squares fit to the training snapshot. The initial state and derivative are taken at
the first snapshot column, whose time is `first(times)` when `times` is given.
"""
function _reduced_system(
        fom::FullOrderModel, snapshot::AbstractMatrix, V::AbstractMatrix, U::AbstractMatrix,
        name::Symbol; times = nothing, kwargs...
    )
    iv = fom.iv
    D = Differential(iv)
    dim = size(V, 2)
    state_name = gensym(:ŷ)
    reduced_state = (@variables $state_name(iv)[1:dim])[1]
    rhs = _reduced_rhs(fom, reduced_state, V, U; kwargs...)

    reduced_snapshot = V' * Matrix{Float64}(snapshot[fom.rows, :])
    lift = Matrix{Float64}(snapshot) / reduced_snapshot
    lift[fom.rows, :] = V
    lift = lift[reduce(vcat, (field.rows for field in fom.fields)), :]
    lift_name = gensym(:lift)
    lift_parameter = (@parameters $lift_name[1:length(fom.source_unknowns), 1:dim])[1]

    replacements = Dict{Any, Any}()
    reconstruction = Equation[]
    offset = 0
    for field in fom.fields
        rows = (offset + 1):(offset + length(field.rows))
        offset = last(rows)
        value = Symbolics.wrap(lift_parameter[rows, :] * reduced_state)
        value = isempty(field.shape) ? only(Symbolics.scalarize(value)) :
            reshape(value, field.shape)
        replacements[field.variable] = value
        push!(reconstruction, Symbolics.wrap(field.variable) ~ value)
    end
    preserved = map(fom.observed) do equation
        equation.lhs ~ substitute(equation.rhs, replacements; kwargs...)
    end

    initial_state = reduced_snapshot[:, 1]
    initial_derivative = _evaluate(
        Symbolics.scalarize(rhs), Symbolics.unwrap.(Symbolics.scalarize(reduced_state)),
        reshape(initial_state, :, 1), fom.parameter_values, iv,
        isnothing(times) ? nothing : times[1:1]; kwargs...
    )
    initial_conditions = Dict{Any, Any}(fom.parameter_values)
    initial_conditions[Symbolics.unwrap(reduced_state)] = initial_state
    initial_conditions[Symbolics.unwrap(D(reduced_state))] = vec(initial_derivative)
    initial_conditions[Symbolics.unwrap(lift_parameter)] = lift

    reduced = System(
        [D(reduced_state) ~ rhs], iv, Symbolics.unwrap.(Symbolics.scalarize(reduced_state)),
        [fom.parameters; Symbolics.unwrap(lift_parameter)];
        name, observed = [reconstruction; preserved], initial_conditions
    )
    for key in (ModelingToolkit.ProblemTypeCtx, ModelingToolkit.MiscSystemData)
        SymbolicUtils.hasmetadata(fom.system, key) || continue
        reduced = SymbolicUtils.setmetadata(
            reduced, key, SymbolicUtils.getmetadata(fom.system, key, nothing)
        )
    end
    return complete(reduced)
end

function _deim(
        sys::System, snapshot::AbstractMatrix, pod_dim::Integer, deim_dim::Integer,
        name::Symbol; snapshot_times = nothing, training_parameters = Dict{Any, Any}(),
        kwargs...
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
    nonlinear_snapshot = _evaluate(
        fom.nonlinear, fom.unknowns, states, fom.parameter_values, fom.iv, snapshot_times;
        kwargs...
    )
    nonlinear_basis = _pod_basis(nonlinear_snapshot, deim_dim)
    return _reduced_system(
        fom, snapshot, state_basis, nonlinear_basis, name; times = snapshot_times, kwargs...
    )
end

"""
    $(FUNCTIONNAME)(
        sys::ModelingToolkit.System,
        snapshot::AbstractMatrix,
        pod_dim::Integer;
        deim_dim::Integer = pod_dim,
        name::Symbol = Symbol(nameof(sys), :_deim),
        snapshot_times = nothing,
        kwargs...
    ) -> ModelingToolkit.System

Reduce a first-order `ModelingToolkit.System` with Proper Orthogonal Decomposition (POD)
and the Discrete Empirical Interpolation Method (DEIM).

The rows of `snapshot` follow `ModelingToolkit.unknowns(sys)` and each column is one
time instance. The state basis is the POD basis of `snapshot`, and the DEIM basis is the
POD basis of the nonlinear terms evaluated at the snapshot columns. Both bases are
computed with [`TSVD`](@ref).

`sys` may contain algebraic equations or array equations. It is structurally simplified
as needed, which may scalarize the equations offline, but the returned system always has
one array differential equation for the reduced state and one reconstruction observed
equation per source field. Full-grid reconstruction coefficients are stored in one array
parameter. Unknowns eliminated during simplification are reconstructed by a least-squares
fit to `snapshot`. Terms that are linear in the unknowns with numeric coefficients are
projected exactly; every other state-dependent term is interpolated by DEIM. Pass
MethodOfLines `symbolic_discretize` systems before structural simplification so every
element of each field is available for reconstruction.

Nonlinear terms are evaluated at the numeric parameter defaults of `sys`. Use the
problem/solution method for problem-specific parameter values. The reduced state and its
derivative at the first snapshot column are stored as initial conditions.

Construct a `DAEProblem` with `build_initializeprob = false` from the returned system
to keep array code generation, or call `ModelingToolkit.mtkcompile` on it and construct
an `ODEProblem` to generate scalar code for the reduced equation. Reconstruct fields with
`SymbolicIndexingInterface.observed(reduced_system, field)`.

# Arguments
- `sys::ModelingToolkit.System`: first-order source system without subsystems.
- `snapshot::AbstractMatrix`: state-by-time snapshot matrix for `sys`.
- `pod_dim::Integer`: number of POD state modes to retain.

# Keywords
- `deim_dim::Integer = pod_dim`: number of DEIM modes for the nonlinear terms.
- `name::Symbol = Symbol(nameof(sys), :_deim)`: name of the reduced system.
- `snapshot_times = nothing`: time of each snapshot column; required when the equations
  depend on the independent variable.
- `kwargs...`: keyword arguments forwarded to `Symbolics.substitute`.

# Returns
- `ModelingToolkit.System`: the completed reduced system.

# Throws
- `DimensionMismatch`: if `snapshot` does not have one row per unknown of `sys`, or if
  `snapshot_times` does not have one entry per column.
- `ArgumentError`: if `sys` does not simplify to an explicit first-order ODE, or if a
  nonlinear term cannot be evaluated numerically.

# Examples
```julia
reduced_system = deim(source_system, snapshots, 4; deim_dim = 6)
```
"""
function deim(
        sys::System, snapshot::AbstractMatrix, pod_dim::Integer;
        deim_dim::Integer = pod_dim, name::Symbol = Symbol(nameof(sys), :_deim),
        snapshot_times = nothing, kwargs...
    )::System
    return _deim(sys, snapshot, pod_dim, deim_dim, name; snapshot_times, kwargs...)
end

const SymbolicProblem = Union{SciMLBase.AbstractODEProblem, SciMLBase.AbstractDAEProblem}

"""
    $(FUNCTIONNAME)(
        prob::Union{SciMLBase.AbstractODEProblem, SciMLBase.AbstractDAEProblem},
        sol,
        pod_dim::Integer;
        deim_dim::Integer = pod_dim,
        name::Union{Nothing, Symbol} = nothing,
        kwargs...
    ) -> ModelingToolkit.System

Reduce the symbolic system behind `prob` with POD-DEIM, using one of its saved solutions
`sol` as the training snapshot.

`prob` must have been constructed from a `ModelingToolkit.System`, such as the array-form
`DAEProblem` returned by MethodOfLines. The saved states are the snapshot columns, the
saved times supply the independent variable for time-dependent terms, and the parameter
values of `prob` are used for training and become the defaults of the reduced system.
`sol` may be the `SciMLBase.PDETimeSeriesSolution` returned by MethodOfLines or its
underlying `SciMLBase.AbstractODESolution`. See the system method for the reduction
itself and for how to construct problems from the returned system.

# Arguments
- `prob`: symbolic first-order problem whose function stores a `ModelingToolkit.System`.
- `sol`: saved solution obtained from `prob`.
- `pod_dim::Integer`: number of POD state modes to retain.

# Keywords
- `deim_dim::Integer = pod_dim`: number of DEIM modes for the nonlinear terms.
- `name::Union{Nothing, Symbol} = nothing`: name of the reduced system. The default appends
  `_deim` to the name of the full system.
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
reduced_system = deim(full_problem, full_solution, 4)
reduced_problem = DAEProblem(
    reduced_system, nothing, full_problem.tspan; build_initializeprob = false
)
```
"""
function deim(
        prob::SymbolicProblem, sol::SciMLBase.AbstractODESolution, pod_dim::Integer;
        deim_dim::Integer = pod_dim, name::Union{Nothing, Symbol} = nothing, kwargs...
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
    reduced_name = isnothing(name) ? Symbol(nameof(sys), :_deim) : name
    return _deim(
        sys, snapshot, pod_dim, deim_dim, reduced_name;
        snapshot_times = sol.t, training_parameters, kwargs...
    )
end

function deim(
        prob::SymbolicProblem, sol::SciMLBase.PDETimeSeriesSolution, pod_dim::Integer;
        kwargs...
    )::System
    return deim(prob, sol.original_sol, pod_dim; kwargs...)
end
