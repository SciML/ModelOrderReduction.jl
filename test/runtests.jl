using SafeTestsets

@safetestset "Quality Assurance" begin include("qa.jl") end
@safetestset "POD" begin include("DataReduction.jl") end
@safetestset "utils" begin include("utils.jl") end
@safetestset "DEIM" begin include("deim.jl") end
