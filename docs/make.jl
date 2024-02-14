push!(LOAD_PATH, "../src/")

using Documenter, ACiD

makedocs(
    sitename = "ACiD.jl",
    modules = [ACiD],
    pages = ["Home" => "index.md"],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
)

deploydocs(; repo = "github.com/jakubpeleska/ACiD.jl")