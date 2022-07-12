#--------- Data Reduction -----------------#

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

@testset "POD-Utils" begin
    solution = lorenz_prob()
    VoV = solution.u
    M = matricize(VoV)
    @test size(M,1) == length(VoV[1]) # Parameters
    @test size(M,2) == length(VoV) # Time
end

@testset "POD - Attractor Test" begin
    sol = lorenz_prob()
    solution = Array(sol)

    order = 2
    solver = SVD()
    reducer = POD(solution,order)
    reduce!(reducer,solver)

    reducer.renergy

    # Ad-hoc tests. To be checked with Chris.
    @test size(reducer.rbasis,2) == reducer.nmodes
    @test size(reducer.rbasis,1) == size(solution,1)
    @test reducer.renergy > 0.9

    order = 2
    solver = TSVD()
    reducer = POD(solution,order)
    reduce!(reducer,solver)

    # Ad-hoc tests. To be checked with Chris.
    @test size(reducer.rbasis,2) == reducer.nmodes
    @test size(reducer.rbasis,1) == size(solution,1)

    order = 2
    solver = RSVD()
    reducer = POD(solution,order)
    reduce!(reducer,solver)


    # Ad-hoc tests. To be checked with Chris.
    @test size(reducer.rbasis,2) == reducer.nmodes
    @test size(reducer.rbasis,1) == size(solution,1)
end
