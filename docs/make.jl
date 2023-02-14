using Documenter, ModelOrderReduction

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

include("pages.jl")

makedocs(sitename = "ModelOrderReduction.jl",
         authors = "Bowen S. Zhu",
         modules = [ModelOrderReduction],
         clean = true, doctest = false, linkcheck = true,
         strict = [
             :doctest,
             :linkcheck,
             :parse_error,
             :example_block,
             # Other available options are
             # :autodocs_block, :cross_references, :docs_block, :eval_block, :example_block, :footnote, :meta_block, :missing_docs, :setup_block
         ],
         format = Documenter.HTML(assets = ["assets/favicon.ico"],
                                  canonical = "https://docs.sciml.ai/ModelOrderReduction/stable/"),
         pages = pages)

deploydocs(repo = "github.com/SciML/ModelOrderReduction.jl.git")
