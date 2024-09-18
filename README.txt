# About pydpac

pydpac contains variational and ensemble-based data assimiliation (DA) algorithms and simple models written in pure Python. Try a jupyter notebook named Demo.ipynb to find out the usage.

# Source code for the submitted article

To try the ensemble variational blending DA in the Nested Lorenz system (Nakashita and Enomoto, submitted to *Tellus A*), checkout `nested_envar` branch and follow the steps below. 

1. run `model/lorenz3m.py` to create a nature run.

2. execute Makefile in `analysis` directory to compile the Fortran-based numerical optimization libraries ([CG+](https://users.iems.northwestern.edu/~nocedal/CG+.html) by G. Liu, J. Nocedal, and R. Waltz, and [LBFGS](http://users.iems.northwestern.edu/~nocedal/lbfgs.html) by J. Nocedal).

3. execute `run_l05nest.sh` in the parent directory.

# Authors

* NAKASHITA, Saori: programmer
* ENOMOTO, Takeshi: project lead

 
