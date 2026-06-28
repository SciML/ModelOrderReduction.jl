using SciMLTesting, ModelOrderReduction, Test

run_qa(
    ModelOrderReduction;
    explicit_imports = true,
    # Whole-package JET (`report_package`/`test_package`) hits a toplevel
    # `invalid redefinition of constant ModelOrderReduction.TSVD` error: the package
    # exports a `TSVD` struct whose name collides with the `TSVD` dependency package
    # under JET's toplevel virtualizer. JET coverage is provided by the targeted
    # `report_call` checks in jet_tests.jl, which sidestep the toplevel re-eval.
    jet = false,
    ei_kwargs = (;
        # `topsort_equations` is owned by ModelingToolkitBase, re-exported but not
        # marked public by ModelingToolkit, and there is no make-public plan for it.
        # `via_owners` flags the re-export; `are_public` flags the non-public status.
        all_qualified_accesses_via_owners = (;
            ignore = (
                :topsort_equations,
            ),
        ),
        all_qualified_accesses_are_public = (;
            ignore = (
                :topsort_equations,
            ),
        ),
    )
)
