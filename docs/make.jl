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
    pages=["Tutorial" => ["Models" => "regression_tutorial.md"], "API" => "index.md"],
)

deploydocs(; repo="github.com/dfcorbin/NonlinearBandits.jl", devbranch="main")
