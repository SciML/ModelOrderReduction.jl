using ModelOrderReduction
using Plots
using OrdinaryDiffEq

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
plot(solution[:,1],solution[:,2],solution[:,3])
# savefig("lorenz_attractor.png")

## Two way POD
reducer = POD(solution,2)
reduce!(reducer,SVD())
bases = reducer.rbasis
plot(bases[:,1],bases[:,2],label="POD2")
# savefig("pod2.png")


## One way POD
reducer = POD(solution,1)
reduce!(reducer,SVD())
bases = reducer.rbasis
plot(bases[:,1],label="POD1")
# savefig("pod1.png")
plot(solution[:,3],label='z')
# savefig("z.png")
