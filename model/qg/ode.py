import numpy as np

def rk4(f, x, dt, *params):
    k1 = dt * f(x, *params)
    k2 = dt * f(x + 0.5*k1, *params)
    k3 = dt * f(x + 0.5*k2, *params)
    k4 = dt * f(x + k3, *params)
    return (k1 + 2*(k2 + k3) + k4) / 6
