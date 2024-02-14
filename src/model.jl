
struct ACiDFluxModel{M}
    m::M
    ∇new::Threads.Atomic{Int64}
    should_run::Threads.Atomic{Bool}
    barrier::Barrier

    ∇new_local::Int64
    ∇next_wait::Real
end

function update!(
    opt::Flux.Optimise.AbstractOptimiser,
    model::ACiDFluxModel,
    gs::Core.Any,
)
    Flux.update!(opt, model.m, gs)

    if model.∇new_local >= model.∇next_wait
        # Wait for 1 averaging step before grad
        wait(model.barrier)
        reset(model.barrier)

        # update the number of grad step to take before the next communication.
        model.∇next_wait += 1 / self.rate_com

        # else
        #     wait(model.barrier)
    end

    Threads.atomic_add!(model.∇new, 1)
    ∇new_local += 1
end
