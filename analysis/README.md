Python interface for minimization methods (lbfgs, cgf) used in `minimize.py` 
are generated with [f2py](https://numpy.org/doc/stable/f2py/index.html).

You need `meson` package for the build backend of `f2py`.

Please see the `Makefile`.
Commands for generating wrappers are as follows:

lbfgs.cpython-<VERSION>-<OS>.so: 
    Fortran source : lbfgs.f
    python3 -m numpy.f2py -c lbfgs.f -m lbfgs

cgf.cpython-<VERSION>-<OS>.so:
    Fortran source : cgsearch.f cgfam.f
    Library dependency : BLAS
    python3 -m numpy.f2py -c cgsearch.f cgfam.f -m cgf -L<LIBRARY PATH FOR BLAS> -lblas
    (!! The order of Fortran sources cannot change !!)

file_utility.cpython-<VERSION>-<OS>.so:
    Fortran source : file_utility.f90
    python3 -m numpy.f2py -c file_utility.f90 -m file_utility
