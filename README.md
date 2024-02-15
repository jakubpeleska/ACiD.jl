# ACiD [![Build Status](https://github.com/jakubpeleska/ACiD.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jakubpeleska/ACiD.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Latest](https://img.shields.io/badge/docs-dev-blue.svg)](https://jakubpeleska.github.io/ACiD.jl/dev/)

This is an implementation of an asynchronous multiprocessing optimization algorithm with a continuous local momentum called **A²CiD²** in the Julia programming language as introduced in [[1]].

> [!NOTE]
> There is also an official demo by the original authors of [[1]] in Python: [AdelNabli/ACiD](https://github.com/AdelNabli/ACiD "AdelNabli/ACiD: Implementation of NeurIPS 2023 paper ACiD: Accelerating Asynchronous Communication in Decentralized Deep Learning.")


## Usage
### Install mpiexecjl

You can install mpiexecjl with MPI.install_mpiexecjl(). The default destination directory is joinpath(DEPOT_PATH\[1\], "bin"), which usually translates to ~/.julia/bin, but check the value on your system. You can also tell MPI.install_mpiexecjl to install to a different directory.

```julia
$ julia
julia> using MPI
julia> MPI.install_mpiexecjl()
```
<!--
### Example
```julia
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

show_output = rank == 0

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
```
-->



To quickly call this wrapper we recommend you to add the destination directory to your PATH environment variable.

## References

\[1\] <span id="[1]">A. Nabli, E. Belilovsky, and E. Oyallon, “**A²CiD²**: Accelerating Asynchronous Communication in Decentralized Deep Learning,” in *Thirty-seventh Conference on Neural Information Processing Systems*, 2023. [Online]. Available: [![arXiv](https://img.shields.io/badge/arXiv-2306.08289-b31b1b.svg)](https://arxiv.org/abs/2306.08289) [![DOI:10.48550/ARXIV.2306.08289](https://img.shields.io/badge/DOI-10.48550/arXiv.2306.08289-b31b1b.svg)](https://doi.org/10.48550/arXiv.2306.08289)</span>

[1]: #[1] "A. Nabli, E. Belilovsky, and E. Oyallon, “A²CiD²: Accelerating Asynchronous Communication in Decentralized Deep Learning,” in Thirty-seventh Conference on Neural Information Processing Systems, 2023."

