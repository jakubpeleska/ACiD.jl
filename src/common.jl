function Init(
    model::M;
    master_rank::Int64 = 0,
    rate_com::Real = 1.0,
    ∇max_steps::Int64 = -1,
) where {M}
    MPI.Init()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    other_rank = Threads.Atomic{Int64}(0)
    ∇new = Threads.Atomic{Int64}(0)
    should_run = Threads.Atomic{Bool}(true)

    barrier_sync_averaging = Barrier(2)
    barrier_end_init = Barrier(2)
    barrier_com_grad = Barrier(2)

    Threads.@spawn sync_process(
        comm,
        master_rank,
        other_rank,
        ∇new,
        barrier_sync_averaging,
    )
    Threads.@spawn gossip_process(
        comm,
        rank,
        other_rank,
        should_run,
        size,
        model,
        barrier_sync_averaging,
        barrier_end_init,
        barrier_com_grad,
        rate_com,
    )

    if rank == master_rank
        Threads.@spawn master_process(
            com,
            size,
            should_run,
            ∇max_steps = ∇max_steps,
        )
    end

    return ACiDFluxModel(model, ∇new, should_run, barrier_com_grad, 0, 0)
end