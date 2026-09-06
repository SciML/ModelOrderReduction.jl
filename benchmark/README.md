Run from the repository root:

```sh
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmark benchmark/array_compilation.jl 8 32 128
```

POD and DEIM dimensions stay fixed at two. The script reports offline reduction,
problem construction, first residual evaluation, and first field reconstruction
separately. Compiler time is Julia's `@timed` compilation counter; total elapsed time
includes work other than compilation. Package loading occurs before the measurements.

To compare fresh-process compilation across grids, invoke the second command once
per size. Passing several sizes measures the first case cold and subsequent cases
with reusable compilation already warm. Equation and observable counts describe the
symbolic representation; timings are empirical evidence, not an asymptotic proof.

Reconstruction uses `SymbolicIndexingInterface.observed`, retaining the whole array
expression. The convenience `getu` path currently encounters
https://github.com/SciML/ModelingToolkit.jl/issues/5072 for these observables.
