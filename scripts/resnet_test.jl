using Flux

using MLUtils, CUDA

using Metalhead

using Optimisers

using MLDatasets: CIFAR10

using ProgressBars



model = ResNet(18; pretrain = false, inchannels = 3, nclasses = 10) |> gpu

optim = Flux.setup(Flux.Adam(), model) |> gpu

labels = collect(0:9)
batchsize = 128

trainSet = CIFAR10()
trainX = trainSet.features
trainY = Flux.onehotbatch(trainSet.targets, labels)
trainLoader = DataLoader(
    (data = trainX, label = trainY),
    batchsize = batchsize,
    shuffle = true,
)

testSet = CIFAR10(Tx = Float64, split = :test)
testX = testSet.features
testY = Flux.onehotbatch(testSet.targets, labels)
testLoader = DataLoader((testX, testY), batchsize = batchsize, shuffle = true)


loss3(m, (x, y)) = Flux.logitcrossentropy(m(x), y) |> gpu

for epoch in 1:100
    for d in ProgressBar(trainLoader)
        d = d |> gpu
        ∂L∂m = Flux.gradient(loss3, model, d...)[1]
        Flux.update!(optim, model, ∂L∂m)
    end

    testLoss = 0
    for (x, y) in testLoader
        x = x |> gpu
        y = y |> gpu
        testLoss += Flux.logitcrossentropy(model(x), y, agg = sum)
    end
    println("Test Loss:", testLoss / length(testSet))
end

