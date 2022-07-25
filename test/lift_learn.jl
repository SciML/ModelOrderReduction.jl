using Test, ModelOrderReduction
using ModelingToolkit, Symbolics

@variables t x(t)
D = Differential(t)

@testset "-eˣ" begin
    eqs = [D(x) ~ -exp(x)]
    @named sys = ODESystem(eqs, t, [x], []; checks = false)
    new_sys = polynomialization(sys)
    # expect
    # dx/dt = -y
    # dy/dt = -y²

    # with
    # y = eˣ
    # dy/dt = (dy/dx)(dx/dt) = (deˣ/dx)(dx/dt) = eˣ(-y) = y(-y) = -y²
    new_eqs = ModelingToolkit.get_eqs(new_sys)
    @test length(new_eqs) == 2
    new_dvs = ModelingToolkit.get_states(new_sys)
    @test length(new_dvs) == 2
    new_rhs = Symbolics.rhss(new_eqs)
    polynomial_dict, residual = semipolynomial_form(new_rhs, new_dvs, 2)
    @test iszero(residual) # check if all equations are polynomials
end

@testset "sin(x)" begin
    eqs = [D(x) ~ sin(x)]
    @named sys = ODESystem(eqs, t, [x], []; checks = false)
    new_sys = polynomialization(sys)
    # expect
    # dx/dt = y₁
    # dy₁/dt = y₁y₂
    # dy₂/dt = -y₁^2

    # with
    # y₁ = sin(x)
    # y₂ = cos(x)
    # dy₁/dt = (dy₁/dx)(dx/dt) = cos(x)y₁ = y₁y₂
    # dy₂/dt = (dy₂/dx)(dx/dt) = -sin(x)y₁ = -y₁²
    new_eqs = ModelingToolkit.get_eqs(new_sys)
    @test length(new_eqs) == 3
    new_dvs = ModelingToolkit.get_states(new_sys)
    @test length(new_dvs) == 3
    new_rhs = Symbolics.rhss(new_eqs)
    polynomial_dict, residual = semipolynomial_form(new_rhs, new_dvs, 2)
    @test iszero(residual) # check if all equations are polynomials
end

@testset "polynomial" begin
    eqs = [D(x) ~ x^2]
    @named sys = ODESystem(eqs, t, [x], []; checks = false)
    new_sys = polynomialization(sys)
    # expect no change
    new_eqs = ModelingToolkit.get_eqs(new_sys)
    @test new_eqs == eqs
    new_dvs = ModelingToolkit.get_states(new_sys)
    @test length(new_dvs) == 1
end

@testset "f^g" begin
    eqs = [D(x) ~ x + sin(x)^tan(x)]
    @named sys = ODESystem(eqs, t, [x], []; checks = false)
    @test_throws ArgumentError polynomialization(sys)
end

@testset "floating-point exponent" begin
    eqs = [D(x) ~ x + x^3.4]
    @named sys = ODESystem(eqs, t, [x], []; checks = false)
    @test_throws ArgumentError polynomialization(sys)
end

@testset "include independent variable" begin
    eqs = [D(x) ~ x + t]
    @named sys = ODESystem(eqs, t, [x], []; checks = false)
    @test_throws ArgumentError polynomialization(sys)
end
