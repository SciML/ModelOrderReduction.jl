using Documenter, ModelOrderReduction

include("pages.jl")

makedocs(sitename = "ModelOrderReduction.jl",
         authors = "Bowen S. Zhu",
         modules = [ModelOrderReduction],
         clean = true, doctest = false,
         format = Documenter.HTML(canonical = "https://docs.sciml.ai/ModelOrderReduction/stable/"),
         pages = pages)

deploydocs(repo = "github.com/SciML/ModelOrderReduction.jl.git")
