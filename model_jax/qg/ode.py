
def rk4(f, x, dt, *params):
    k1 = dt * f(x, *params)[0]
    k2 = dt * f(x + 0.5*k1, *params)[0]
    k3 = dt * f(x + 0.5*k2, *params)[0]
    k4 = dt * f(x + k3, *params)[0]
    return (k1 + 2*(k2 + k3) + k4) / 6
