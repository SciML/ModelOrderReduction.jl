# ModelOrderReduction development

- High-level `deim` must preserve array-valued reduced dynamics and array-valued field
  reconstruction. Keep full-grid reconstruction coefficients in array parameters;
  scalar reconstruction equations make the generated graph grow with the grid.
- Compilation complexity claims apply at fixed reduced dimensions, field count, count of
  distinct symbolic forcing expressions, and nonlinear stencil size. Offline reduction, problem construction, reconstruction
  storage, and field evaluation can still scale with the full-order dimension.
- Validate changes with `GROUP=Core julia --project -e 'using Pkg; Pkg.test()'` and
  `GROUP=QA julia --project -e 'using Pkg; Pkg.test()'`. Build documentation with
  `julia --project=docs docs/make.jl` when changing public behavior or examples.
- Array-form regression coverage must check both generated graph size across grids
  and numerical dynamics/reconstruction. Equation counts alone do not establish
  numerical correctness or constant compilation cost.
- Term separation goes through `Symbolics.semilinear_form`. Only linear terms with numeric
  coefficients are projected exactly; every other state-dependent term, including linear
  terms with parameter or time-dependent coefficients, is DEIM-sampled. Keep it that way
  unless an affine parameter decomposition is implemented, because symbolic coefficients in
  the projected matrix make the reduced graph grow with the grid.
- Source systems must list scalar unknowns (scalarized array elements or `mtkcompile`
  output). A `System` built without explicit unknowns collects both an array variable and
  its indexed elements, which no problem constructor accepts either.
- Both `DAEProblem(rom, nothing, tspan; build_initializeprob = false)` and
  `ODEProblem(mtkcompile(rom), nothing, tspan; build_initializeprob = false)` are supported
  and tested. Reconstruct fields with `SymbolicIndexingInterface.observed`; the `sol[field]`
  convenience path hits https://github.com/SciML/ModelingToolkit.jl/issues/5072.
- The reduced equation is array linear algebra: projected matrices, the stencil rows of the
  state basis, and the reconstruction coefficients are array parameters, and only the
  DEIM-sampled nonlinear terms are scalar expressions. Numeric matrices embedded as literals
  in a sum of array terms break `mtkcompile` (SymbolicUtils sorts them with `isless`), so
  keep them as parameters.
