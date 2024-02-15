using Flux

using MLUtils

using Metalhead

using Optimisers

using MLDatasets: CIFAR10

using ProgressBars: ProgressBar, update, set_description, set_postfix

using Printf

using ACiD, MPI


function test_cifar_training()
    epochs = 1

    model_resnet = ResNet(18; pretrain = false, inchannels = 3, nclasses = 10)  # Changed to model_resnet because of unnecessary retyping.

    model = ACiD.Init(model_resnet)

    optim = Flux.setup(Flux.Adam(), model.m)

    labels = collect(UInt32, 0:9)  # UInt32 supposed to be the same type as lables in the CIFAR10 dataset
    batchsize = 128

    trainSet = CIFAR10(Tx = Float32, split = :train)  # Writing this explicitly might be more readable.
    trainX = trainSet.features
    trainY = Flux.onehotbatch(trainSet.targets, labels)
    trainLoader = DataLoader(
        (data = trainX, label = trainY),
        batchsize = batchsize,
        shuffle = true,
    )

    testSet = CIFAR10(Tx = Float32, split = :test)  # HotFix: Tx = Float64 causes retyping in Flux model during inference
    testX = testSet.features
    testY = Flux.onehotbatch(testSet.targets, labels)
    testLoader =
        DataLoader((testX, testY), batchsize = batchsize, shuffle = true)

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    show_output = rank == 0

    testAcc::Float64 = 0.0
    testLoss::Float32 = 0.0
    trainLoss::Float32 = 0.0

    for e in 1:epochs
        trainLoss = 0.0
        trainedSamples::Int64 = 0
        trainIter = show_output ? ProgressBar(trainLoader) : trainLoader
        for (x, y) in trainIter
            l::Float32, gs = Flux.withgradient(model.m::ResNet) do m
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

        testSamples::Int64 = 0
        testAcc = 0.0
        testLoss = 0.0
        for (x, y) in testLoader
            testSamples += size(x, 4)
            out::Matrix{Float32} = model.m(x)
            ŷ::Vector{UInt32} = Flux.onecold(out, labels)
            yₜ::Vector{UInt32} = Flux.onecold(y, labels)
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

    return Float32[trainLoss, Float32(testAcc), testLoss]
end



