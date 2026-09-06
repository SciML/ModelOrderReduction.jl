# Developer Interface

The supported user workflow is deliberately small: construct a [`POD`](@ref) from
state snapshots, then call [`reduce!`](@ref) with one of the built-in backends,
[`SVD`](@ref), [`TSVD`](@ref), or [`RSVD`](@ref). The public API does not require users
to interact with the abstract types described below.

## POD reduction contract

`POD` accepts either an `AbstractMatrix{T}` whose columns are state snapshots or an
`AbstractVector{<:AbstractVector{T}}` containing state snapshots, with `T <: AbstractFloat`.
All snapshots must describe the same state dimension. The positional `nmodes` form fixes
the retained rank. The keyword form selects the rank from `min_renergy`, `min_nmodes`, and
`max_nmodes`.

The reduction contract is:

1. Construct `pod = POD(snapshots, nmodes)` or `POD(snapshots; kwargs...)`.
2. Call `reduce!(pod, SVD())`, `reduce!(pod, TSVD())`, or `reduce!(pod, RSVD())`.
3. Read `pod.rbasis`, `pod.spectrum`, `pod.nmodes`, and `pod.renergy` after the call.

`reduce!` mutates `pod` and returns `nothing`. The three concrete backends are the
supported choices; their keyword arguments are forwarded to their respective SVD
implementations as described in the API reference.

```jldoctest
julia> using ModelOrderReduction

julia> snapshots = [1.0 0.0; 0.0 1.0];

julia> pod = POD(snapshots, 1);

julia> reduce!(pod, SVD()); pod.nmodes
1
```

## Reduction pipeline

[`deim`](@ref) reduces a system in three internal steps. The source system is first
brought into the explicit first-order form described by
[`ModelOrderReduction.FullOrderModel`](@ref), whose right-hand sides are split by
[`ModelOrderReduction.separate_terms`](@ref) into a numeric linear part, forcing terms,
and the state-dependent terms handled by interpolation. The POD state basis and the DEIM
basis are then computed from the snapshot and from the interpolated terms evaluated at the
snapshot. Finally, the reduced system is assembled with one array differential equation
for the reduced state and one reconstruction equation per
[`ModelOrderReduction.SourceField`](@ref). These types and functions are internal and
may change without notice.

```@docs
ModelOrderReduction.FullOrderModel
ModelOrderReduction.SourceField
ModelOrderReduction.separate_terms
```

## Internal type hierarchy

The following abstract types are implementation details. They are documented so that
contributors can understand the source hierarchy, but they are not exported and are not
stable extension points:

```@docs
ModelOrderReduction.AbstractReductionProblem
ModelOrderReduction.AbstractDRProblem
ModelOrderReduction.AbstractSVD
```

Do not subtype these types, dispatch on them from downstream packages, or add new
`reduce!` methods for custom algorithm types. Such methods would depend on internal
dispatch and field invariants that are not covered by the public API. If a supported
extension point is needed, it should first be designed and documented as a separate
interface with generic tests.
