using SafeTestsets

@safetestset "POD" begin include("DataReduction.jl") end
@safetestset "DEIM" begin include("deim.jl") end
