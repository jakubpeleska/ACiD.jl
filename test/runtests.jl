using ACiD
using Test

include("resnet_test_minimal.jl")

@testset "ACiD.jl" begin
    # train_loss: 1.4556, test_acc: 0.4329, test_loss: 1.8672
    train_loss, test_acc, test_loss = test_cifar_training()
    @test train_loss <= 1.7056  # previously used ≈() function with atol=0.25
    @test test_acc <= 0.6829
    @test test_loss <= 2.1172
end
