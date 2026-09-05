# ModelOrderReduction development

- High-level `deim` must preserve array-valued reduced dynamics and array-valued field
  reconstruction. Keep full-grid reconstruction coefficients in array parameters;
  scalar reconstruction equations make the generated graph grow with the grid.
- Compilation complexity claims apply at fixed reduced dimensions, field count, forcing
  expression count, and nonlinear stencil size. Offline reduction, problem construction, reconstruction
  storage, and field evaluation can still scale with the full-order dimension.
- Validate changes with `GROUP=Core julia --project -e 'using Pkg; Pkg.test()'` and
  `GROUP=QA julia --project -e 'using Pkg; Pkg.test()'`. Build documentation with
  `julia --project=docs docs/make.jl` when changing public behavior or examples.
- Array-form regression coverage must check both generated graph size across grids
  and numerical dynamics/reconstruction. Equation counts alone do not establish
  numerical correctness or constant compilation cost.
