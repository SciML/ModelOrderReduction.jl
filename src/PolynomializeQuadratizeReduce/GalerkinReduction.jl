using ModelingToolkit
using OrdinaryDiffEq
using Symbolics
using LinearAlgebra
using ModelOrderReduction

unwrap(x) = Symbolics.value(x)

function differentiated_variable(eq)
    lhs = unwrap(eq.lhs)
    args = arguments(lhs)
    return Num(args[1])
end

function ordered_rhs(sys)
    eqs = equations(sys)
    xs = unknowns(sys)

    rhs_by_var = Dict{Any, Any}()

    for eq in eqs
        x = differentiated_variable(eq)
        rhs_by_var[x] = eq.rhs
        rhs_by_var[unwrap(x)] = eq.rhs
    end

    return [rhs_by_var[x] for x in xs]
end

"""
    galerkin_project_system_affine(sys, V, xbar, a_vars; pmap=Dict(), name=:rom)

Project an explicit ODE system onto the affine trial space

    x(t) ≈ xbar + V*a(t).

Here `x(t)` is the full state vector of `sys`, `xbar` is a fixed offset vector,
`V` is an `n × r` basis matrix, and `a(t)` is the reduced state vector with
entries `a_vars`.

The reduced system is constructed by direct Galerkin projection:

    a'(t) = V' * f(xbar + V*a(t)),

where `f` is the right-hand side of the full system. The returned system also
contains observed equations reconstructing the full state variables as

    x_i(t) ~ xbar[i] + sum(V[i, α] * a_α(t) for α in 1:r).

Arguments:
- `sys`: ModelingToolkit ODE system.
- `V`: projection basis of size `(n, r)`, where `n = length(unknowns(sys))`.
- `xbar`: affine offset vector of length `n`.
- `a_vars`: reduced state variables of length `r`.

Keywords:
- `pmap`: optional parameter substitutions applied before projection.
- `name`: name of the returned reduced system.

Returns:
- A ModelingToolkit `System` for the affine Galerkin reduced-order model.
"""
function galerkin_project_system_affine(sys, V, xbar, a_vars; pmap = Dict(), name = :rom)
    xs = unknowns(sys)

    n = length(xs)
    n == size(V, 1) || error("V must have size (n,r), where n = length(unknowns(sys))")
    length(xbar) == n || error("xbar must have length n")
    r = size(V, 2)
    length(a_vars) == r || error("length(a_vars) must equal size(V,2)")

    iv = ModelingToolkit.get_iv(sys)
    Dred = Differential(iv)

    rhs = unwrap.(ordered_rhs(sys))

    if !isempty(pmap)
        subdict = Dict(Symbolics.unwrap(k) => v for (k, v) in pmap)
        rhs = Symbolics.substitute.(rhs, Ref(subdict))
    end

    x_subs = Dict{Any, Any}()

    for i in 1:n
        rec = xbar[i]

        for α in 1:r
            rec += V[i, α] * a_vars[α]
        end

        x_subs[xs[i]] = rec
        x_subs[unwrap(xs[i])] = rec
    end

    f_affine = [
        Symbolics.substitute(rhs[i], x_subs)
            for i in 1:n
    ]

    rhs_red = Vector{Any}(undef, r)

    for α in 1:r
        expr = zero(Num)

        for i in 1:n
            expr += V[i, α] * f_affine[i]
        end

        rhs_red[α] = Symbolics.simplify(Symbolics.expand(expr))
    end

    eqs_red = [Dred(a_vars[α]) ~ rhs_red[α] for α in 1:r]

    obs = Vector{Equation}(undef, n)

    for i in 1:n
        rec = xbar[i]

        for α in 1:r
            rec += V[i, α] * a_vars[α]
        end

        obs[i] = xs[i] ~ Symbolics.simplify(rec)
    end

    return System(eqs_red, iv; observed = obs, name = name)
end

"""
    galerkin_project_system(sys, V, a_vars; pmap=Dict(), name=:rom)

Project an explicit ODE system onto the linear trial space

    x(t) ≈ V*a(t).

This is the zero-offset special case of `galerkin_project_system_affine`, namely
`xbar = zeros(n)`, where `n = length(unknowns(sys))`.

Arguments:
- `sys`: ModelingToolkit ODE system.
- `V`: projection basis of size `(n, r)`.
- `a_vars`: reduced state variables of length `r`.

Keywords:
- `pmap`: optional parameter substitutions applied before projection.
- `name`: name of the returned reduced system.

Returns:
- A ModelingToolkit `System` for the linear Galerkin reduced-order model.
"""
function galerkin_project_system(sys, V, a_vars; pmap = Dict(), name = :rom)
    n = length(unknowns(sys))
    xbar = zeros(Float64, n)

    return galerkin_project_system_affine(
        sys,
        V,
        xbar,
        a_vars;
        pmap = pmap,
        name = name,
    )
end
