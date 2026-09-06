using ModelOrderReduction, ModelingToolkit, OrdinaryDiffEq
import SymbolicIndexingInterface as SII
@independent_variables t
D = Differential(t)
function trial(n)
    @variables z(t)[1:n]
    states = Symbolics.unwrap.(Symbolics.scalarize(z))
    sys = System([D(z) ~ -z - z .^ 3], t, states, []; name = :scaling)
    snapshot = [sin(0.13 * i * j) + cos(0.07 * i * (j + 1)) for i in 1:n, j in 1:8]
    offline = @timed deim(sys, snapshot, 2)
    rom = offline.value
    construction = @timed DAEProblem(rom, nothing, (0.0, 0.1); build_initializeprob = false)
    prob = construction.value
    du = zeros(2); residual = zeros(2)
    rhs = @timed prob.f(residual, du, prob.u0, prob.p, 0.0)
    reconstruction = @timed SII.observed(prob, z)(prob.u0, prob.p, first(prob.tspan))
    println(
        (
            n = n, equations = length(equations(rom)), observables = length(observed(rom)),
            offline_seconds = offline.time, construction_seconds = construction.time,
            construction_compile_seconds = construction.compile_time,
            rhs_seconds = rhs.time, rhs_compile_seconds = rhs.compile_time,
            reconstruction_seconds = reconstruction.time,
            reconstruction_compile_seconds = reconstruction.compile_time,
        )
    )
    return flush(stdout)
end
for n in parse.(Int, ARGS)
    trial(n)
end
