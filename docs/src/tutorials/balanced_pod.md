# Balanced POD

[`bpod`](@ref) reduces a continuous-time linear system
```math
\dot x = A x + B u, \qquad y = C x + D u
```
from primal and adjoint impulse-response snapshots. The construction is the one in
Rowley, *International Journal of Bifurcation and Chaos* (2005). It approximates
square-root balanced truncation. [`baltrunc`](@ref) solves the two Gramian Lyapunov
equations directly and is the appropriate method when those solves are affordable.
[`bpod`](@ref) is the snapshot form of the same truncation: the caller supplies the
factors, and the function returns a [`BalancedTruncation`](@ref).

The snapshot factors must already contain the quadrature weights. If `w_k` is the
quadrature weight at time `t_k`, the corresponding column of `X` is
`sqrt(w_k) * exp(A t_k) b_j` and the corresponding column of `Y` is
`sqrt(w_k) * exp(A' t_k) c_i`. The empirical Gramians are then `X X'` and `Y Y'`.
[`bpod`](@ref) does not form those products and does not generate the snapshots.
Applying the square-root weights a second time squares them.

The product `Y' * X` is a dense matrix with one row per adjoint snapshot and one
column per primal snapshot. Decomposing it dominates the cost after the factors
exist. Sparse, diagonal, or banded `A`, `B`, and `C` do not change that product.

A finite horizon omits the tail of the impulse response, and the quadrature rule
adds its own error. Both appear in the approximate Hankel singular values.
Finite-horizon balanced POD does not generally make the reduced model stable: `A`
can be Hurwitz while the reduced state matrix is not. When a stable reduced model
is required, inspect the eigenvalues of that matrix. If the original system is
unstable, the finite-horizon singular values are not the infinite-horizon Hankel
singular values of a stable system. When the retained factors span the state, the
full-order projector is a similarity and can reproduce the input-output map even
though the Hankel values are still inaccurate. Agreement of that full-order
reduction is therefore not evidence that the singular values have converged. The
example below keeps two modes and compares that reduced map with the original
system.

## A stable system

The eigenvalues of `A` are `-1`, `-2`, `-3`, and `-6`, so both the primal and
adjoint impulses decay. The slowest factor is `exp(-t)`. Over a horizon of length
16 that factor is about `1e-7`, so the horizon covers the decay of these impulses.
The step `Δt = 1/16` is a separate approximation. The feedthrough `D` is copied
unchanged.

```@example balanced_pod
using LinearAlgebra
using ModelOrderReduction

A = [
    -1.0 20.0 0.0 0.0
    0.0 -2.0 15.0 0.0
    0.0 0.0 -3.0 10.0
    0.0 0.0 0.0 -6.0
]
B = [
    1.0 0.0
    0.4 1.0
    0.0 0.5
    0.2 -0.3
]
C = [
    1.0 0.2 0.0 0.4
    0.0 1.0 0.3 -0.2
]
D = [
    0.15 -0.02
    0.01 0.08
]
eigvals(A)
```

## Weighted impulses

The time grid is uniform, with `T = 16` and `Δt = 1/16`. On a uniform grid the
composite trapezoidal rule assigns weight `Δt` to each interior node and weight
`Δt / 2` to the endpoints. Those weights sum to the horizon. The square root is
applied once, inside the snapshot matrix.

`exp(A t)` is a dense exponential. It is a clear way to build this four-state
example, and it is not a large-scale snapshot algorithm. A large or sparse system
should advance the primal impulses `exp(A t) b_j` and the adjoint impulses
`exp(A' t) c_i` with a matrix-free exponential or an ODE solver, attach the same
square-root weights, and pass the resulting factors to [`bpod`](@ref).

```@example balanced_pod
t = collect(range(0.0, 16.0; length = 257))
dt = t[2] - t[1]
weights = fill(dt, length(t))
weights[1] /= 2
weights[end] /= 2

function weighted_impulses(generator, columns, t, weights)
    n, m = size(columns)
    nt = length(t)
    X = zeros(n, m * nt)
    for (k, tk) in pairs(t)
        response = exp(generator * tk)
        scale = sqrt(weights[k])
        for j in 1:m
            X[:, (j - 1) * nt + k] = scale * (response * columns[:, j])
        end
    end
    return X
end

X = weighted_impulses(A, B, t, weights)
Y = weighted_impulses(A', C', t, weights)
(size(X), sum(weights))
```

## Reduced model

With the default relative tolerance, this system retains every positive value,
because the smallest is still larger than one thousandth of the largest. Requesting
`n = 2` drops the two smaller values. The result is a [`BalancedTruncation`](@ref):
`rom.A`, `rom.B`, `rom.C`, and `rom.D` are the reduced system, `rom.hsv` contains
only the retained approximate Hankel singular values, `rom.Tr` reconstructs the
original state, and `rom.S` reduces it. The identity `rom.S * rom.Tr ≈ I` is the
biorthogonality of the projectors. It holds for a poor snapshot grid as well, so it
does not measure the accuracy of `rom.hsv`.

A distinct singular value determines its left and right singular vectors up to
one shared sign. A repeated singular value determines only the subspace, which
may be rotated. Negating column `j` of `rom.Tr` and row `j` of `rom.S` together
changes the signs in `rom.A`, `rom.B`, and `rom.C` and leaves the input-output
map unchanged.

```@example balanced_pod
spectrum = bpod(A, B, C, X, Y, D; n = 4, atol = 0.0, rtol = 0.0)
rom = bpod(A, B, C, X, Y, D; n = 2)
spectrum.hsv, rom.hsv, opnorm(rom.S * rom.Tr - I(2)), rom.D == D
```

## Output comparison

For a constant input the state is `x(t) = A^{-1} (exp(A t) - I) B u` when `t > 0`.
The function below evaluates that output and the frequency response
`C (iω I - A)^{-1} B + D`. The comparison uses five output times and five
frequencies. It is a sample of the two maps, not a certified `H∞` norm.

```@example balanced_pod
function step_output(A, B, C, D, u, t)
    x = t == 0 ? zeros(size(A, 1)) : A \ (exp(A * t) * (B * u) - B * u)
    return C * x + D * u
end

function frequency_response(A, B, C, D, ω)
    return C * (((im * ω) * I(size(A, 1)) - A) \ B) + D
end

u = [1.0, -0.4]
times = (0.0, 0.5, 1.0, 2.0, 5.0)
frequencies = (0.1, 0.7, 1.0, 1.3, 10.0)
step_gap = maximum(times) do t
    norm(step_output(A, B, C, D, u, t) - step_output(rom.A, rom.B, rom.C, rom.D, u, t))
end
frequency_gap = maximum(frequencies) do ω
    opnorm(
        frequency_response(A, B, C, D, ω) -
            frequency_response(rom.A, rom.B, rom.C, rom.D, ω)
    )
end
step_gap, frequency_gap
```

The printed gaps are samples of the difference between the original system and the
order-2 model. They combine the two discarded Hankel modes with the remaining
horizon and quadrature error, and they are not a bound at every frequency or input.
A coarser grid, such as `T = 2` and `Δt = 1`, leaves the projectors biorthogonal
while moving the approximate Hankel values, and with them this reduced response.
Refining the snapshot grid makes the model approach [`baltrunc`](@ref). It does
not remove the error of the order that was requested.

[`baltrunc`](@ref) computes this same order from the infinite-horizon Gramians.
The first number below is its sampled distance from the plant. The second is the
sampled distance between that truncation and the snapshot model. On this grid the
two reduced models are close, and both remain separated from the plant by the two
discarded modes. Neither number is an `H∞` bound.

```@example balanced_pod
reference = baltrunc(A, B, C, D; n = 2)
truncation_gap = maximum(frequencies) do ω
    opnorm(
        frequency_response(A, B, C, D, ω) -
            frequency_response(reference.A, reference.B, reference.C, reference.D, ω)
    )
end
snapshot_gap = maximum(frequencies) do ω
    opnorm(
        frequency_response(reference.A, reference.B, reference.C, reference.D, ω) -
            frequency_response(rom.A, rom.B, rom.C, rom.D, ω)
    )
end
truncation_gap, snapshot_gap
```
