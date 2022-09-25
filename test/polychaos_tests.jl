using Test
using ModelOrderReduction, Symbolics, PolyChaos
MO = ModelOrderReduction

# testing extraction of independent variables 
@variables t, z, u(t), v(t)[1:4], w(t,z), x(t,z)[1:4]

@test isequal(MO.get_independent_vars(u), [t])
@test isequal(MO.get_independent_vars(v[1]), [t])
@test isequal(MO.get_independent_vars(v[2]), [t])
@test isequal(MO.get_independent_vars(w),[t,z])
@test isequal(MO.get_independent_vars(x[2]),[t,z])
@test isequal(MO.get_independent_vars(collect(v)), [[t] for i in 1:length(v)])
@test isequal(MO.get_independent_vars(collect(x)), [[t,z] for i in 1:length(v)])

# testing PCE expansion generation
@parameters a
@variables t, x(t)
D = Differential(t)
test_equation = [D(x) ~ a*x]
