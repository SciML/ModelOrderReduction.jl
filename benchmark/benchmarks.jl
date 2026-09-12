using ModelOrderReduction, BenchmarkTools
using ModelingToolkit

const SUITE = BenchmarkGroup()

@independent_variables t
@variables u(t)[1:4]
D = Differential(t)
snapshot = [
    sin(0.3 * i * j) + cos(0.2 * i * (j + 1)) for i in 1:4, j in 1:8
]

@mtkcompile sys = System([D(u) ~ -u - u .^ 3], t)
csys = complete(sys)

# =============================================================================
# DEIM reduction (symbolic system + snapshot matrix)
# =============================================================================

SUITE["deim"] = BenchmarkGroup()

SUITE["deim"]["deim_2"] = @benchmarkable deim($csys, $snapshot, 2)
SUITE["deim"]["deim_3"] = @benchmarkable deim($csys, $snapshot, 3)

# =============================================================================
# POD-Galerkin reduction
# =============================================================================

SUITE["pod"] = BenchmarkGroup()

SUITE["pod"]["pod_2"] = @benchmarkable pod($csys, $snapshot, 2)
SUITE["pod"]["pod_3"] = @benchmarkable pod($csys, $snapshot, 3)
