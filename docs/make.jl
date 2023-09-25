using Documenter, ModelOrderReduction

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

include("pages.jl")

makedocs(sitename = "ModelOrderReduction.jl",
         authors = "Bowen S. Zhu",
         modules = [ModelOrderReduction],
         clean = true, doctest = false, linkcheck = true,
         warnonly = [:missing_docs, :example_block],
         format = Documenter.HTML(assets = ["assets/favicon.ico"],
                                  canonical = "https://docs.sciml.ai/ModelOrderReduction/stable/"),
         pages = pages)

deploydocs(repo = "github.com/SciML/ModelOrderReduction.jl.git")
