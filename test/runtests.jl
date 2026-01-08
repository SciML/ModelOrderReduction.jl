using SafeTestsets

@safetestset "Quality Assurance" begin
    include("qa.jl")
end
@safetestset "Explicit Imports" begin
    include("explicit_imports.jl")
end
@safetestset "POD" begin
    include("DataReduction.jl")
end
@safetestset "SOD" begin
    include("SOD.jl")
end
@safetestset "utils" begin
    include("utils.jl")
end
@safetestset "DEIM" begin
    include("deim.jl")
end
