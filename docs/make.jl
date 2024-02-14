push!(LOAD_PATH, "../src/")

using Documenter, ACiD

installation = ["installation_mpiexecjl.md"]

resources = ["api.md", "p2p_sync.md", "p2p_averaging.md"]

makedocs(
    sitename = "ACiD.jl",
    modules = [ACiD],
    pages = [
        "Home" => "index.md",
        "Installation" => installation,
        "Resources" => resources,
    ],
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
)

deploydocs(; repo = "github.com/jakubpeleska/ACiD.jl")