using Flux

using MLUtils

using Metalhead

using Optimisers

using MLDatasets: CIFAR10

using ProgressBars: ProgressBar, update, set_description, set_postfix

using Printf

using ACiD, MPI


model = ResNet(18; pretrain = false, inchannels = 3, nclasses = 10)

model = ACiD.Init(model)

optim = Flux.setup(Flux.Adam(), model.m)

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

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

show_output = rank == 1

epochs = 100
for e in 1:epochs
    trainLoss = 0
    trainedSamples = 0
    trainIter = show_output ? ProgressBar(trainLoader) : trainLoader
    for (x, y) in trainIter
        l, gs = Flux.withgradient(model.m) do m
            Flux.logitcrossentropy(m(x), y, agg = sum)
        end
        ACiD.update!(optim, model, gs[1])

        trainedSamples += size(x, 4)
        trainLoss += l
        if show_output
            set_description(trainIter, @sprintf("Epoch %d/%d", e, epochs))
            set_postfix(
                trainIter,
                train_loss = @sprintf("%.4f", trainLoss / trainedSamples)
            )
        end
    end
    trainLoss /= trainedSamples

    testSamples = 0
    testAcc = 0
    testLoss = 0
    for (x, y) in testLoader
        testSamples += size(x, 4)
        out = model.m(x)
        ŷ = Flux.onecold(out, labels)
        yₜ = Flux.onecold(y, labels)
        testAcc += sum(ŷ .== yₜ)
        testLoss += Flux.logitcrossentropy(out, y, agg = sum)
    end

    testAcc /= testSamples
    testLoss /= testSamples

    if show_output
        set_postfix(
            trainIter,
            train_loss = @sprintf("%.4f", trainLoss),
            test_acc = @sprintf("%.4f", testAcc),
            test_loss = @sprintf("%.4f", testLoss)
        )
        update(trainIter, 0, force_print = true)
    end
end

