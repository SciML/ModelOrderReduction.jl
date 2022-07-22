using SafeTestsets

@safetestset "POD" begin include("pod.jl") end
@safetestset "Lift & Learn" begin include("lift_learn.jl") end
