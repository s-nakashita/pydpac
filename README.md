# About pydpac

**pydpac** contains variational and ensemble-based data assimiliation (DA) algorithms and simple models written in pure Python. 

Try `Demo.ipynb` for an example with L96.

`main.py` describes the experimental parameters and provides modules for conducting an observation system simulation experiment (OSSE) along with `exp_func.py`. 

`run.sh` provides the procedures to compare several DA algorithms.

See `analysis/README.md` if you would like to use Fortran-based numerical optimization algorithms ([CG+](https://users.iems.northwestern.edu/~nocedal/CG+.html) by G. Liu, J. Nocedal, and R. Waltz, and [LBFGS](http://users.iems.northwestern.edu/~nocedal/lbfgs.html) by J. Nocedal).

# Available DA algorithms

- Deterministic DA
    * Kalman filter ([Kalman 1960](https://doi.org/10.1115/1.3662552))
    * Three-dimensional, Four-dimensional variational method (3DVar, 4DVar; [Talagrand and Courtier 1987](https://doi.org/10.1002/qj.49711347812))
    * (For nesting system) Three-dimensional variational blending assimilation ([Guidard and Fischer 2008](https://doi.org/10.1002/qj.215))
- Ensemble DA
    * Ensemble Kalman Filter ([Evensen 1994](https://doi.org/10.1029/94JC00572))
        + Ensemble transform Kalman filter (ETKF; [Bishop et al. 2001](https://doi.org/10.1175/1520-0493%282001%29129%3C0420:ASWTET%3E2.0.CO;2))
        + Perturbed observation method (PO; [Burgers et al. 1998](https://doi.org/10.1175/1520-0493%281998%29126%3C1719:ASITEK%3E2.0.CO;2), [Houtekamer et al.2005](https://doi.org/10.1175/MWR-2864.1))
        + Serial ensemble square root filter (EnSRF; [Whitaker and Hamill 2002](https://doi.org/10.1175/1520-0493%282002%29130%3C1913:EDAWPO%3E2.0.CO;2))
        + Ensemble adjustment Kalman filter (EAKF, local least-square formulation; [Anderson 2003](https://doi.org/10.1175/1520-0493%282003%29131<0634:ALLSFF>2.0.CO;2))
        + Local ensemble transform Kalman filter (LETKF; [Hunt et al. 2007](https://doi.org/10.1016/j.physd.2006.11.008))
    * Maximum likelihood ensemble filter (MLEF; [Zupanski 2005](https://doi.org/10.1175/MWR2946.1), [Zupanski et al. 2008](https://doi.org/10.1002/qj.251))
    * Ensemble variational method (EnVar; [Liu et al. 2008](https://doi.org/10.1175/2008MWR2312.1))
    * (For nesting system) ensemble variational blending assimilation ([Nakashita and Enomoto 2025](https://doi.org/10.16993/tellusa.4089))

# Forecast models

- `lorenz.py`: A spatially one-dimensional chaotic model ([Lorenz 1995](https://www.ecmwf.int/node/10829);[Lorenz and Emanuel 1998](https://doi.org/10.1175/1520-0469%281998%29055<0399:OSFSWO>2.0.CO;2))

- `lorenz2.py`: A spatially one-dimensional chaotic model with large-scale dynamics (type II model in [Lorenz 2005](https://doi.org/10.1175/JAS3430.1))

- `lorenz3.py`: A spatially one-dimensional chaotic model with scale interaction (type III model in [Lorenz 2005](https://doi.org/10.1175/JAS3430.1))

- `burgers.py`: A spatially one-dimensional advection and diffusion model ([Burgers 1948](https://doi.org/10.1016/S0065-2156%2808%2970100-5);[Zupanski et al. 2008](https://doi.org/10.1002/qj.251))

- `kdvb.py`: A one-dimensional Korteweg&ndash;de Vries&ndash;Burgers (KdVB) model ([Marchant and Smyth 2002](https://doi.org/10.1098/rspa.2001.0868);[Zupanski 2005](https://doi.org/10.1175/MWR2946.1);[Enomoto and Nakashita 2024](https://doi.org/10.16993/tellusa.3255))

<!---
- `qgmain.py`: A two-dimensional quasi-geostrophic model ([Sakov and Oke 2008](https://doi.org/10.1111/j.1600-0870.2007.00299.x);[Enomoto and Nakashita 2022 (in Japanese)](https://www.dpri.kyoto-u.ac.jp/nenpo/no65/ronbunB/a65b0p12.pdf))
--->

# Source code for the submitted article

To try the ensemble variational blending DA in the nested Lorenz system (Nakashita and Enomoto 2025, *Tellus A*), checkout `v1.0.0` and follow the steps below. 

1. Run `model/lorenz3m.py` to create a nature run.

2. Run Makefile in `analysis` directory to compile the Fortran-based numerical optimization libraries.

3. Run `run_l05nest.sh` in the parent directory.

# Authors

* NAKASHITA, Saori: programmer
* ENOMOTO, Takeshi: project lead

