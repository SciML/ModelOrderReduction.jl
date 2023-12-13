using ModelOrderReduction, Aqua
@testset "Aqua" begin
    #Aqua.find_persistent_tasks_deps(ModelOrderReduction)
    Aqua.test_ambiguities(ModelOrderReduction, recursive = false)
    Aqua.test_deps_compat(ModelOrderReduction)
    Aqua.test_piracies(ModelOrderReduction,
        treat_as_own = [])
    Aqua.test_project_extras(ModelOrderReduction)
    Aqua.test_stale_deps(ModelOrderReduction)
    Aqua.test_unbound_args(ModelOrderReduction)
    Aqua.test_undefined_exports(ModelOrderReduction)
end
