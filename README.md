# ACiD [![Build Status](https://github.com/jakubpeleska/ACiD.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jakubpeleska/ACiD.jl/actions/workflows/CI.yml?query=branch%3Amain)

This is an implementation of an asynchronous multiprocessing optimization algorithm with a continuous local momentum called **A²CiD²** in the Julia programming language as introduced in [[1]].

> [!NOTE]
> There is also an official demo by the original authors of [[1]] in Python: [AdelNabli/ACiD](https://github.com/AdelNabli/ACiD "AdelNabli/ACiD: Implementation of NeurIPS 2023 paper ACiD: Accelerating Asynchronous Communication in Decentralized Deep Learning.")


## Usage
### Install mpiexecjl

You can install mpiexecjl with MPI.install_mpiexecjl(). The default destination directory is joinpath(DEPOT_PATH[1], "bin"), which usually translates to ~/.julia/bin, but check the value on your system. You can also tell MPI.install_mpiexecjl to install to a different directory.

```julia
$ julia
julia> using MPI
julia> MPI.install_mpiexecjl()
```

### Example
```julia
s = "Julia syntax highlighting";
println(s);
```




To quickly call this wrapper we recommend you to add the destination directory to your PATH environment variable.

## References

\[1\] <span id="[1]">A. Nabli, E. Belilovsky, and E. Oyallon, “**A²CiD²**: Accelerating Asynchronous Communication in Decentralized Deep Learning,” in *Thirty-seventh Conference on Neural Information Processing Systems*, 2023. [Online]. Available: [![arXiv](https://img.shields.io/badge/arXiv-2306.08289-b31b1b.svg)](https://arxiv.org/abs/2306.08289) [![DOI:10.48550/ARXIV.2306.08289](https://img.shields.io/badge/DOI-10.48550/arXiv.2306.08289-b31b1b.svg)](https://doi.org/10.48550/arXiv.2306.08289)</span>

[1]: #[1] "A. Nabli, E. Belilovsky, and E. Oyallon, “A²CiD²: Accelerating Asynchronous Communication in Decentralized Deep Learning,” in Thirty-seventh Conference on Neural Information Processing Systems, 2023."

