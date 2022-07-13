using Symbolics, ModelingToolkit

function deim_interpolation_indices(basis::AbstractMatrix)::Vector{Int}
    dim = size(basis, 2)
    indices = Vector{Int}(undef, dim)
    indices[1] = argmax(abs.(@view basis[:, 1]))
    for l in 2:dim
        U = @view basis[:, 1:(l - 1)]
        P = @view indices[1:(l - 1)]
        PᵀU = @view U[P, :]
        uₗ = @view basis[:, l]
        Pᵀuₗ = @view uₗ[P, :]
        c = PᵀU \ Pᵀuₗ
        r = vec(uₗ - U * c)
        indices[l] = argmax(abs.(r))
    end
    return indices
end

function deim(sys::ODESystem, pod_basis::AbstractMatrix;
              deim_basis::AbstractMatrix = pod_basis,
              deim_dim::Integer = size(pod_basis, 2),
              name::Symbol = Symbol(nameof(sys), "_deim"))::ODESystem
    rhs = [eq.rhs for eq in equations(sys)]
    # rhs = A * vars + F
    # A is the coefficient matrix of the linear part
    # F is the remaining nonlinear vector function
    A, F = semilinear_form(rhs, states(sys))
    U = @view deim_basis[:, 1:deim_dim] # DEIM projection basis
    indices = deim_interpolation_indices(U) # DEIM interpolation indices
    t = ModelingToolkit.get_iv(sys) # the single independent variable
    D = Differential(t)
    pod_dim = size(pod_basis, 2) # the dimension of POD basis
    @variables y_pod[1:pod_dim](t) # new variables from POD reduction
    pod_projection = Symbolics.scalarize(states(sys) .~ pod_basis * y_pod)
    pod_reduction = Dict(eq.lhs => eq.rhs for eq in pod_projection) # POD reduction dict
    # the DEIM projector (not DEIM basis) satisfies
    # F(original_vars) ≈ projector * F(pod_basis * reduced_vars)[indices]
    projector = ((@view U[indices, :])' \ U')'
    deim_nonlinear = map(f -> substitute(f, pod_reduction), F[indices])
    deim_nonlinear = projector * deim_nonlinear # DEIM approximation for nonlinear func F
    A_pod = pod_basis' * A * pod_basis
    ODESystem(D.(y_pod) .~ A_pod * y_pod + pod_basis' * deim_nonlinear;
              observed = [observed(sys); pod_projection], name = name)
end

export deim
