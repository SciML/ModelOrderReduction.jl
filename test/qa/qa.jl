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
        # Internal symbolic-stack names that ModelingToolkit/Symbolics re-export but
        # neither own nor mark public. `via_owners` flags the re-export; `are_public`
        # flags the non-public status. ModelOrderReduction uses them as internal API.
        all_qualified_accesses_via_owners = (;
            ignore = (
                # owner ModelingToolkitBase, accessed via ModelingToolkit
                :get_eqs, :get_initial_conditions, :get_iv, :get_observed,
                :get_unknowns, :get_var_to_name, :topsort_equations,
                # owner SymbolicUtils, accessed via Symbolics
                :scalarize, :unwrap,
                # owner SymbolicIndexingInterface, accessed via Symbolics
                :getname,
            ),
        ),
        all_qualified_accesses_are_public = (;
            ignore = (
                # not public in ModelingToolkit
                :get_eqs, :get_initial_conditions, :get_iv, :get_observed,
                :get_unknowns, :get_var_to_name, :topsort_equations,
                # not public in Symbolics
                :getname, :scalarize, :unwrap, :rhss, :value,
                # not public in SymbolicUtils
                :isadd, :ismul,
            ),
        ),
    )
)
