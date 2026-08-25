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

`sys` is assumed to have no internal systems. End users are encouraged to call
`ModelingToolkit.mtkcompile` beforehand.

The POD basis used for DEIM interpolation is obtained from the snapshot matrix of the
nonlinear terms, which is computed by executing the runtime-generated function for
nonlinear expressions.

# Arguments
- `sys::ModelingToolkit.ODESystem`: compiled first-order ODE system without internal
  subsystems.
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
"""
function deim(
        sys::ODESystem, snapshot::AbstractMatrix, pod_dim::Integer;
        deim_dim::Integer = pod_dim, name::Symbol = Symbol(nameof(sys), :_deim),
        kwargs...
    )::ODESystem
    return _deim_impl(
        sys, snapshot, pod_dim; deim_dim, name, snapshot_times = nothing, kwargs...
    )
end

function _deim_impl(
        sys::ODESystem, snapshot::AbstractMatrix, pod_dim::Integer;
        deim_dim::Integer, name::Symbol, snapshot_times, kwargs...
    )::ODESystem
    sys = deepcopy(sys)
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

    nonlinear_snapshot = if isnothing(snapshot_times)
        # Generate an in-place function from the symbolic nonlinear expressions for the
        # existing explicit-system API.
        F_func! = build_function(F, dvs; expression = Val{false}, kwargs...)[2]
        values = similar(snapshot)
        for i in axes(snapshot, 2)
            F_func!(view(values, :, i), view(snapshot, :, i))
        end
        values
    else
        # The DAE entry point deliberately avoids generating a full-order scalar function.
        _evaluate_symbolic_snapshot(F, dvs, snapshot, iv, snapshot_times)
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

    # Replace full-order initialization data with the projected initial state. Array-form
    # systems can retain parent-array guesses that are not keyed by the scalarized `dvs`.
    @set! sys.guesses = Dict{Any, Any}()
    @set! sys.initialization_eqs = Equation[]
    reduced_initial = V' * view(snapshot, :, 1)
    @set! sys.initial_conditions = Dict(Symbolics.unwrap(ŷ) => reduced_initial)

    return complete(sys)
end

"""
    $(FUNCTIONNAME)(
        prob::SciMLBase.DAEProblem,
        sol,
        pod_dim::Integer;
        deim_dim::Integer = pod_dim,
        name::Union{Nothing, Symbol} = nothing,
        kwargs...
    ) -> ModelingToolkit.ODESystem

Reduce a first-order symbolic `DAEProblem` and one of its saved solutions using POD-DEIM.

This method is intended for array-form problems such as those produced by MethodOfLines v1.
The full-order solve retains its array-form DAE compilation path. For the symbolic reduction,
ModelingToolkit tearing eliminates algebraic unknowns and isolates the differential equations
without constructing or compiling a full-order `ODEProblem`. Only the reduced ODE is compiled
when the returned system is used to construct a problem.

`sol` may be the `SciMLBase.PDETimeSeriesSolution` returned by MethodOfLines or its underlying
`SciMLBase.AbstractODESolution`.

# Arguments
- `prob::SciMLBase.DAEProblem`: symbolic first-order DAE problem whose function stores its
  ModelingToolkit system in `prob.f.sys`.
- `sol`: saved solution obtained from `prob`.
- `pod_dim::Integer`: number of POD state modes to retain.

# Keywords
- `deim_dim::Integer = pod_dim`: number of DEIM modes for nonlinear terms.
- `name::Union{Nothing, Symbol} = nothing`: name assigned to the reduced system. The default
  appends `_deim` to the full system's name.
- `kwargs...`: keyword arguments forwarded to ModelingToolkit transformations and symbolic
  substitutions.

# Examples
```julia
full_problem = discretize(pde_system, discretization; fallback = false)
full_solution = solve(full_problem)
reduced_system = deim(full_problem, full_solution, 4)
```

# Returns
- `ModelingToolkit.ODESystem`: the reduced and completed explicit ODE system.

# Throws
- `ArgumentError`: if `sol` is not from `prob`, does not start at the beginning of the problem,
  or the DAE cannot be torn into an explicit ODE whose unknowns match the saved states.
- `DimensionMismatch`: if the saved solution does not match the DAE system.
"""
function deim(
        prob::SciMLBase.DAEProblem, sol::SciMLBase.AbstractODESolution,
        pod_dim::Integer; deim_dim::Integer = pod_dim,
        name::Union{Nothing, Symbol} = nothing, kwargs...
    )::ODESystem
    hasproperty(prob.f, :sys) && prob.f.sys isa ODESystem ||
        throw(ArgumentError("the DAE problem must contain a symbolic ModelingToolkit system"))
    raw_sys = prob.f.sys
    hasproperty(sol, :prob) && hasproperty(sol.prob.f, :sys) && sol.prob.f.sys === raw_sys ||
        throw(ArgumentError("the solution must have been obtained from the supplied DAE problem"))
    first(sol.t) == first(prob.tspan) ||
        throw(ArgumentError("the solution must save the state at the start of the DAE problem"))
    raw_variables = ModelingToolkit.get_unknowns(raw_sys)
    full_snapshot = Array(sol)
    size(full_snapshot, 1) == length(raw_variables) ||
        throw(DimensionMismatch("solution states must match the DAE system unknowns"))

    sys = complete(tearing(raw_sys))
    deqs, residual_equations = get_deqs(sys)
    isempty(residual_equations) ||
        throw(ArgumentError("tearing the DAE must produce an explicit ODE system"))
    length(deqs) == length(ModelingToolkit.get_unknowns(sys)) ||
        throw(ArgumentError("the torn DAE system must have one differential equation per unknown"))
    raw_indices = Dict(
        Symbolics.unwrap(variable) => index
            for (index, variable) in enumerate(raw_variables)
    )
    snapshot_rows = map(ModelingToolkit.get_unknowns(sys)) do variable
        index = get(raw_indices, Symbolics.unwrap(variable), nothing)
        isnothing(index) &&
            throw(
            ArgumentError(
                "a differential unknown produced by tearing is absent from the DAE state vector"
            )
        )
        index
    end
    snapshot = full_snapshot[snapshot_rows, :]
    reduced_name = isnothing(name) ? Symbol(nameof(raw_sys), :_deim) : name
    return _deim_impl(
        sys, snapshot, pod_dim;
        deim_dim, name = reduced_name, snapshot_times = sol.t, kwargs...
    )
end

function deim(
        prob::SciMLBase.DAEProblem, sol::SciMLBase.PDETimeSeriesSolution,
        pod_dim::Integer; kwargs...
    )::ODESystem
    return deim(prob, sol.original_sol, pod_dim; kwargs...)
end
