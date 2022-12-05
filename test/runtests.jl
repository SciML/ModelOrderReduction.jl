using SafeTestsets

@time @safetestset "POD" begin include("DataReduction.jl") end
@time @safetestset "utils" begin include("utils.jl") end
@time @safetestset "DEIM" begin include("deim.jl") end
@time @safetestset "PCE" begin include("pce.jl") end
