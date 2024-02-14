module ACiD

using MPI

using Flux, Optimisers
using ExponentialUtilities

include("p2p_averaging.jl")
include("p2p_sync.jl")
include("model.jl")


function Init(root::Int64 = 0)
    MPI.Init()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    Threads.@spawn sync_process(rank, size, comm)
    Threads.@spawn gossip_process(rank)

    if rank == root
        Threads.@spawn master_process(size)
    end
end

end
