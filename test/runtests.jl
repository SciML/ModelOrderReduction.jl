using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

@time begin
    if GROUP == "All" || GROUP == "DEIM"
        @time @safetestset "POD" begin include("DataReduction.jl") end
        @time @safetestset "utils" begin include("utils.jl") end
        @time @safetestset "DEIM" begin include("deim.jl") end
    end
    if GROUP == "All" || GROUP == "PCE"
        @time @safetestset "PCE" begin include("pce.jl") end
    end
end
