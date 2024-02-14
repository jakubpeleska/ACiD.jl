# Install mpiexecjl

You can install mpiexecjl with MPI.install\_mpiexecjl(). The default destination directory is joinpath(DEPOT\_PATH\[1\], "bin"), which usually translates to ~/.julia/bin, but check the value on your system. You can also tell MPI.install\_mpiexecjl to install to a different directory.

Enter Julia and write:
```julia
$ julia
julia> using MPI
julia> MPI.install_mpiexecjl()
```