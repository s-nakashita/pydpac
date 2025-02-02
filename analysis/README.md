# DA algorithms

For default parameter settings, see `config/config_XXX_sample.py`. 

Naming rules: 

- XXX = \[4d\]\<DA type\>\[loc type\]

- The prefix '4d' means 4-dimensional DA.

- DA types

    * kf: `kf.py`
    * var: `var.py`, `var4d.py`
    * eakf, etkf, letkf, po, srf: `enkf.py`, `enkf4d.py`
    * mlef: `mlef.py`, `mlef4d.py`, `lmlef.py`, 'lmlef4d.py`
    * envar: `envar.py`
    * (for nesting system) var_nest: `var_nest.py`
    * (for nesting system) envar_nest: `envar_nest.py`

- localization types

    * 'be': state-space localization with eigen value decomposition
    * 'bm': state-space localization with modulation ensemble ([Bishop et al. 2017](https://doi.org/10.1175/MWR-D-17-0102.1))
    * 'kloc': observation-space localization applied to Kalman gain
    * 'y, cw': observation-space localization for variational assimilation ([Yokota et al. 2016](https://doi.org/10.2151/sola.2016-019))

# Fortran-based numerical optimization libraries

Python interface for minimization methods (lbfgs, cgf) used in `minimize.py` 
are generated with [f2py](https://numpy.org/doc/stable/f2py/index.html).

You need `meson` package for the build backend of `f2py`.

Please see the `Makefile`.
Commands for generating wrappers are as follows:

* lbfgs.cpython-\<PYVERSION\>-\<OS\>.so: 
    Fortran source : lbfgs.f
    python3 -m numpy.f2py -c lbfgs.f -m lbfgs

* cgf.cpython-\<PYVERSION\>-\<OS\>.so:
    Fortran source : cgsearch.f cgfam.f
    Library dependency : BLAS
    python3 -m numpy.f2py -c cgsearch.f cgfam.f -m cgf -L\<LIBRARY PATH FOR BLAS\> -lblas
    (!! The order of Fortran sources must not be changed !!)

* file_utility.cpython-\<PYVERSION\>-\<OS\>.so:
    Fortran source : file_utility.f90
    python3 -m numpy.f2py -c file_utility.f90 -m file_utility
