using Flux, Plots
data = [([x], 2x - x^3) for x in -2:0.1f0:2]

model = Chain(
    Dense(1 => 10, tanh),
    Dense(10 => 10, tanh),
    Dense(10 => 1; bias = false),
    only,
)

optim = Flux.setup(Adam(), model)
for epoch in 1:1000
    Flux.train!((m, x, y) -> (m(x) - y)^2, model, data, optim)
end

plot(x -> 2x - x^3, -2, 2; legend = false)
scatter!(x -> model([x]), -2:0.1f0:2)
