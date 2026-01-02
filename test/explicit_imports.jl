using Test
using ExplicitImports
using ModelOrderReduction

@testset "ExplicitImports" begin
    @testset "No implicit imports" begin
        @test check_no_implicit_imports(ModelOrderReduction) === nothing
    end

    @testset "No stale explicit imports" begin
        @test check_no_stale_explicit_imports(ModelOrderReduction) === nothing
    end
end
