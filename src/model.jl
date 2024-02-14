
struct ACiDFluxModel{M}
    m::M
    ∇new::Threads.Atomic{Int64}
    should_run::Threads.Atomic{Bool}
    barrier::Barrier
    ∇count_local::Threads.Atomic{Int64}
    ∇next_wait::Threads.Atomic{Float64}
end

function update!(opt, model::ACiDFluxModel, gs::Core.Any)
    Flux.update!(opt, model.m, gs)

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    if rank != 0 && model.∇count_local >= model.∇next_wait
        # Wait for 1 averaging step before grad
        wait(model.barrier)
        reset(model.barrier)

        # update the number of grad step to take before the next communication.
        model.∇next_wait += 1 / self.rate_com

        # else
        #     wait(model.barrier)
    end

    Threads.atomic_add!(model.∇new, 1)
    Threads.atomic_add!(model.∇count_local, 1)
end
