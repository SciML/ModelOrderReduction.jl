using Documenter, ModelOrderReduction

include("pages.jl")

makedocs(sitename = "ModelOrderReduction.jl",
         pages = pages)

deploydocs(repo = "github.com/SciML/ModelOrderReduction.jl.git")
