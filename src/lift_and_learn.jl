"""
    lift_and_learn(sys::ODESystem, snapshots::AbstractMatrix, dt::Real, 
                   lifting_map::Function, r::Int; 
                   name::Symbol = Symbol(nameof(sys), :_lnl))

Perform model order reduction using the Lift & Learn method.

# Arguments
- `sys`: The original `ModelingToolkit.ODESystem`.
- `snapshots`: The snapshot matrix of the *original* states.
- `dt`: The sampling time step.
- `lifting_map`: A function `f(u) -> w` mapping original states to lifted states.
- `r`: The dimension of the reduced-order model.
- `name`: (Optional) The name of the resulting system.

# Returns
- A `ModelingToolkit.System`.
"""
function lift_and_learn(sys::ODESystem, snapshots::AbstractMatrix, dt::Real,
        lifting_map::Function, r::Integer;
        name::Symbol = Symbol(nameof(sys), :_lnl))
    # 1. Lifting
    # Apply lifting map column-wise: w = T(x)
    W = mapslices(lifting_map, snapshots, dims = 1)

    # 2. Derivative Estimation (Central Difference)
    # W_dot has 2 fewer columns than W because we lose the endpoints
    W_dot = (W[:, 3:end] .- W[:, 1:(end - 2)]) ./ (2 * dt)

    # Trim W to match the size of W_dot (remove first and last columns)
    W_data = W[:, 2:(end - 1)]

    # 3. POD Basis Construction
    # Leveraging existing POD functionality
    pod_problem = POD(W_data, r)
    reduce!(pod_problem, TSVD())
    Vr = pod_problem.rbasis

    # Project to reduced linear coordinates
    w_r = Vr' * W_data      # (r, k)
    w_r_dot = Vr' * W_dot   # (r, k)

    # 4. Operator Inference (Least Squares)
    k_samples = size(w_r, 2)

    # Prepare data matrix D = [w_r^T, (w_r ⊗ w_r)^T, 1]
    D_linear = w_r'

    # Construct quadratic terms (Full Kronecker product)
    # Note: For high-performance applications, consider compact Kronecker forms.
    D_quad = zeros(eltype(w_r), k_samples, r^2)
    for i in 1:k_samples
        w_curr = w_r[:, i]
        D_quad[i, :] = kron(w_curr, w_curr)
    end

    D_const = ones(eltype(w_r), k_samples, 1)

    # Concatenate to form the full design matrix
    D = hcat(D_linear, D_quad, D_const)

    # Target matrix
    R = w_r_dot'

    # Solve D * O^T = R  =>  O^T = D \ R  =>  O = (D \ R)'
    Operators_T = D \ R
    Operators = Operators_T'

    # Extract operators from the solution matrix
    # Structure of Operators: [A (r columns) | H (r^2 columns) | C (1 column)]
    A_hat = Operators[:, 1:r]
    H_hat = Operators[:, (r + 1):(r + r ^ 2)]
    C_hat = Operators[:, end]

    # 5. Symbolic Reconstruction
    return build_quadratic_system(sys, A_hat, H_hat, C_hat, r, name)
end

"""
    build_quadratic_system(sys, A, H, C, r, name)

Helper function to construct a symbolic ODESystem from learned matrix operators.
"""
function build_quadratic_system(sys, A, H, C, r, name)
    iv = ModelingToolkit.get_iv(sys)
    D = Differential(iv)

    # Define reduced state variables w_1, ..., w_r
    @variables w(iv)[1:r]
    w_vec = collect(w)

    # Construct the quadratic dynamics: dw/dt = Aw + H(w⊗w) + C
    quad_term = kron(w_vec, w_vec)
    rhs = A * w_vec + H * quad_term + C

    eqs = [D(w_vec[i]) ~ rhs[i] for i in 1:r]

    return System(eqs, iv, w_vec, []; name)
end
