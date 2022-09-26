using ModelOrderReduction
using Test
using OrdinaryDiffEq

@testset "ModelOrderReduction.jl" begin
end
include("polychaos_tests.jl")
include("DataReduction.jl")
#---------- Model Reduction ----------------#
