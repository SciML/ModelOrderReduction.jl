using Test, ModelOrderReduction, ModelingToolkit

@independent_variables t
@variables u(t)[1:4]
D = Differential(t)
snapshot = [sin(0.3 * i * j) + cos(0.2 * i * (j + 1)) for i in 1:4, j in 1:8]

@testset "array ROM regressions" begin
    @testset "structure" begin
        @mtkcompile sys = System([D(u) ~ -u - u .^ 3], t)
        rom = deim(sys, snapshot, 2)
        @test length(equations(rom)) == 1
        @test length(observed(rom)) == 1
    end
    @testset "parameter defaults" begin
        @parameters α = 2.0
        @mtkcompile sys = System([D(u) ~ -u - α * u .^ 3], t)
        rom = deim(sys, snapshot, 2)
        default_value = initial_conditions(rom)[Symbolics.unwrap(α)]
        @test SymbolicUtils.unwrap_const(Symbolics.unwrap(default_value)) == 2.0
    end
    @testset "grid-independent graph with uniform forcing" begin
        tree_size(expression) = (
            value = Symbolics.unwrap(expression);
            SymbolicUtils.iscall(value) ?
                1 + sum(tree_size, SymbolicUtils.arguments(value); init = 0) : 1
        )
        function graph_size(n)
            @variables q(t)[1:n]
            @parameters α = 2.0
            @mtkcompile sys = System([D(q) ~ -q - α * q .^ 3 .+ sin(t)], t)
            local_snapshot = [sin(0.3 * i * j) + cos(0.2 * i * (j + 1)) for i in 1:n, j in 1:8]
            rom = deim(sys, local_snapshot, 2; snapshot_times = collect(0.0:0.1:0.7))
            equation = only(equations(rom))
            return tree_size(equation.lhs) + tree_size(equation.rhs)
        end
        @test graph_size(4) == graph_size(12)
    end
    @testset "scalar input" begin
        @variables z₁(t) z₂(t)
        @mtkcompile sys = System([D(z₁) ~ z₁ - z₁^3, D(z₂) ~ -z₂], t)
        rom = deim(sys, [1.0 0.8 0.6; 0.5 0.4 0.3], 1)
        @test length(unknowns(rom)) == 1
        @test isempty(guesses(rom))
        @test isempty(initialization_equations(rom))
    end

end
