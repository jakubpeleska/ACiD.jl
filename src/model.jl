
struct ACiDFluxModel{M}
    m::M
end

function update!(
    opt::Flux.Optimise.AbstractOptimiser,
    model::ACiDFluxModel,
    gs::Core.Any,
)
    Flux.update!(opt, model.m, gs)

    # TODO: call sync
end
