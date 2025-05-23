import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import ode
from math import exp, log, pi

# Reactor and catalyst geometry parameters
d_c = 0.825e-3  # Catalyst particle diameter [m]
d_r = 1e-3      # Reactor diameter [m]
length = 0.26   # Reactor length [m]
eps_c = 0.308   # Catalyst void fraction
eps_r = 0.54    # Reactor void fraction
rho_s = 770     # Solid density [kg/m^3]

# Volume and effective density calculation
v_c = pi * d_c**3 / 6
v_r = pi * d_r**2 * length
p_eff = (rho_s * length * (eps_r + (1 - eps_r) * eps_c) * (1 - eps_c) * v_c) / (v_r * d_c)

# Kinetic models
def get_k1(u, theta, Tref_C=105):
    Tref = Tref_C + 273.15
    T = u[0] + 273.15
    return exp(-theta[0] - theta[1] * 1e4 / 8.314 * (1 / T - 1 / Tref))

def kinetic_model(x, u, theta, model_type):
    k1 = get_k1(u, theta)
    x0, x1, x2, x3 = x
    denom = 1
    if model_type >= 1:
        denom += theta[2] * x2
    if model_type >= 2:
        denom += theta[3] * x1
    if model_type >= 3:
        denom += theta[4] * x0 + theta[5] * x3
    r1 = p_eff * k1 * x0 * x1 / denom**2
    return [-r1, -r1, r1, r1]

# CSTR and PFR model equations
def CSTR_equations(x_out, tau, x_in, u, theta, model_type):
    ra = kinetic_model(x_out, u, theta, model_type)
    return [x_out[i] - x_in[i] - tau * ra[i] for i in range(4)]

def PFR_equations(t, x, u, theta, model_type):
    return kinetic_model(x, u, theta, model_type)

# Solvers
def solve_CSTR(tau, x_in, u, theta, model_type):
    return fsolve(CSTR_equations, [1, 1, 1, 1], args=(tau, x_in, u, theta, model_type))

def solve_PFR(tau, x_in, u, theta, model_type):
    r = ode(PFR_equations).set_integrator('vode', method='bdf')
    r.set_initial_value(x_in, 0).set_f_params(u, theta, model_type)
    return r.integrate(r.t + tau)

# CSTR in series
def CSTR_series(tau, x_in, u, theta, n_stages, model_type):
    x_vals = np.zeros((n_stages + 1, 4))
    x_vals[0, :] = x_in
    stage_tau = tau / n_stages
    for i in range(n_stages):
        x_vals[i + 1, :] = solve_CSTR(stage_tau, x_vals[i, :], u, theta, model_type)
    return x_vals[-1, :]

# Dispatcher
def model_solve(tau, x_in, u, theta, reactor_type, model_type):
    if reactor_type == 0:  # CSTR
        return solve_CSTR(tau, x_in, u, theta, model_type)
    elif reactor_type == 1:  # PFR
        return solve_PFR(tau, x_in, u, theta, model_type)
    else:
        raise ValueError("Invalid reactor type. Use 0 for CSTR or 1 for PFR.")

