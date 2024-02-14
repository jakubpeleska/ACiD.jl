module ACiD

using MPI, Flux
using Revise

include("barrier.jl")
include("common.jl")
include("model.jl")
include("p2p_averaging.jl")
include("p2p_sync.jl")

export Init, update!

export ACiDFluxModel

end
