using NonlinearBandits
using Documenter

DocMeta.setdocmeta!(
    NonlinearBandits, :DocTestSetup, :(using NonlinearBandits); recursive=true
)

makedocs(;
    modules=[NonlinearBandits],
    authors="Doug Corbin <dfcorbin98@gmail.com> and contributors",
    repo="https://github.com/dfcorbin/NonlinearBandits.jl/blob/{commit}{path}#{line}",
    sitename="NonlinearBandits.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://dfcorbin.github.io/NonlinearBandits.jl",
        assets=String[],
    ),
    pages=[
        "Introduction" => "index.md",
        "Tutorial" => ["Partitioned Polynomials" => "ppm_tutorial.md"],
        "API" => ["Bandits" => ["bandits_api.md"], "Models" => ["model_api.md"]],
    ],
)

deploydocs(; repo="github.com/dfcorbin/NonlinearBandits.jl", devbranch="main")
