using SafeTestsets

@safetestset "POD" begin include("pod.jl") end
@safetestset "DEIM" begin include("deim.jl") end
