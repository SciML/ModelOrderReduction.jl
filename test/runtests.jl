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

if GROUP == "All" || GROUP == "QA"
    using Pkg
    Pkg.activate(@__DIR__() * "/qa")
    Pkg.develop(path = dirname(@__DIR__))
    Pkg.instantiate()

    @safetestset "Quality Assurance" begin
        include("qa/qa_tests.jl")
    end
    @safetestset "Explicit Imports" begin
        include("qa/explicit_imports_tests.jl")
    end
    @safetestset "JET Static Analysis" begin
        include("qa/jet_tests.jl")
    end
end
