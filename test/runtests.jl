using ModelOrderReduction
using Test
using OrdinaryDiffEq


#--------- Data Reduction -----------------#
@testset "PCA" begin
    function lorenz_prob()
        function lorenz!(du,u,p,t)
            du[1] = p[1]*(u[2]-u[1])
            du[2] = u[1]*(p[2]-u[3]) - u[2]
            du[3] = u[1]*u[2] - p[3]*u[3]
        end

        u0 = [1,0,0]
        p = [10,28,8/3]
        tspan = (0,100)
        prob = ODEProblem(lorenz!,u0,tspan,p)
        sol = solve(prob,Tsit5())
        sol
    end

    sol = lorenz_prob()
    solution = Matrix(reduce(hcat,sol.u)')

    order = 2
    reducer = PCA(solution,order)
    reduce!(reducer)

    # Ad-hoc tests. To be checked with Chris.
    @test length(reducer.rbasis) == reducer.nmodes
    @test length(reducer.rbasis[1]) == size(solution,2)
    @test reducer.energy > 0.9
end

#---------- Model Reduction ----------------#
