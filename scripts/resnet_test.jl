using Flux

using MLUtils, CUDA

using Metalhead

using Optimisers

using MLDatasets: CIFAR10

using ProgressBars: ProgressBar, update, set_description, set_postfix

using Printf



model = ResNet(18; pretrain = false, inchannels = 3, nclasses = 10)

optim = Flux.setup(Flux.Adam(), model)

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

epochs = 100
for e in 1:epochs
    trainLoss = 0
    trainedSamples = 0
    trainIter = ProgressBar(trainLoader)
    for (x, y) in trainIter
        l, gs = Flux.withgradient(model) do m
            Flux.logitcrossentropy(m(x), y, agg = sum)
        end
        Flux.update!(optim, model, gs[1])

        trainedSamples += size(x, 4)
        trainLoss += l
        set_description(trainIter, @sprintf("Epoch %d/%d", e, epochs))
        set_postfix(
            trainIter,
            train_loss = @sprintf("%.4f", trainLoss / trainedSamples)
        )
    end
    trainLoss /= trainedSamples

    testSamples = 0
    testAcc = 0
    testLoss = 0
    for (x, y) in testLoader
        testSamples += size(x, 4)
        out = model(x)
        ŷ = Flux.onecold(out, labels)
        yₜ = Flux.onecold(y, labels)
        testAcc += sum(ŷ .== yₜ)
        testLoss += Flux.logitcrossentropy(out, y, agg = sum)
    end

    testAcc /= testSamples
    testLoss /= testSamples

    set_postfix(
        trainIter,
        train_loss = @sprintf("%.4f", trainLoss),
        test_acc = @sprintf("%.4f", testAcc),
        test_loss = @sprintf("%.4f", testLoss)
    )
    update(trainIter, 0, force_print = true)
end

