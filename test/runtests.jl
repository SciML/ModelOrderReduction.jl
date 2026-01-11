using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "All" || GROUP == "Core"
    @safetestset "Quality Assurance" begin
        include("qa.jl")
    end
    @safetestset "Explicit Imports" begin
        include("explicit_imports.jl")
    end
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
    @safetestset "JET Static Analysis" begin
        include("nopre/jet_tests.jl")
    end
end
