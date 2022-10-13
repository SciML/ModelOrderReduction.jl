using ModelOrderReduction

#---------- Model Reduction ----------------#
using SafeTestsets
@safetestset "POD" begin include("DataReduction.jl") end
@safetestset "utils" begin include("utils.jl") end
@safetestset "DEIM" begin include("deim.jl") end
@safetestset "PCE" begin include("PCETests.jl") end
