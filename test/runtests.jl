using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "All" || GROUP == "Core"
    @safetestset "POD" begin
        include("DataReduction.jl")
    end
    @safetestset "utils" begin
        include("utils.jl")
    end
    @safetestset "DEIM" begin
        include("deim.jl")
    end
end

if GROUP == "nopre"
    using Pkg
    Pkg.activate(@__DIR__() * "/nopre")
    Pkg.instantiate()

    @safetestset "Quality Assurance" begin
        include("nopre/qa_tests.jl")
    end
    @safetestset "Explicit Imports" begin
        include("nopre/explicit_imports_tests.jl")
    end
    @safetestset "JET Static Analysis" begin
        include("nopre/jet_tests.jl")
    end
end
