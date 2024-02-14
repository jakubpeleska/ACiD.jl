# ACiD.jl

This is an implementation of an asynchronous multiprocessing optimization algorithm with a continuous local momentum called **A²CiD²** in the Julia programming language as introduced in [^1].

!!! note
    There is also an official demo by the original authors of [^1] in Python: [AdelNabli/ACiD](https://github.com/AdelNabli/ACiD "AdelNabli/ACiD: Implementation of NeurIPS 2023 paper ACiD: Accelerating Asynchronous Communication in Decentralized Deep Learning.").
    This code was also used as a reference for our implementation.

## Installation
```@contents
Pages = Main.installation
Depth = 1:2
```
## Resources
```@contents
Pages = Main.resources
Depth = 1:2
```

## Index

```@index
Pages = ["api.md", "p2p_sync.md", "p2p_averaging.md"]
```

[^1]: A. Nabli, E. Belilovsky, and E. Oyallon, “**A²CiD²**: Accelerating Asynchronous Communication in Decentralized Deep Learning,” in *Thirty-seventh Conference on Neural Information Processing Systems*, 2023. [Online]. Available: [![arXiv](https://img.shields.io/badge/arXiv-2306.08289-b31b1b.svg)](https://arxiv.org/abs/2306.08289) [![DOI:10.48550/ARXIV.2306.08289](https://img.shields.io/badge/DOI-10.48550/arXiv.2306.08289-b31b1b.svg)](https://doi.org/10.48550/arXiv.2306.08289)