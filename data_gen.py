import numpy as np
from pyDOE2 import lhs
import pandas as pd
from math import log
from reaction_reactor import model_solve

def generate_parameter_samples(p_samples, pm, ps, method, seed=3):
    """Generate parameter samples using LHS or Gaussian perturbation."""
    if method == 0:
        ci = 2  # 95% confidence interval
        p_mean = np.array(pm)
        p_sd = np.array(ps)

        pb_upper = p_mean + ci * p_sd
        pb_lower = p_mean - ci * p_sd
        bounds = np.vstack((pb_lower, pb_upper)).T

        n_params = bounds.shape[0]
        lhs_points = lhs(n_params, samples=p_samples, criterion='maximin', iterations=50, random_state=seed)
        lhs_samples = np.zeros((p_samples, n_params))

        for i in range(n_params):
            lhs_samples[:, i] = lhs_points[:, i] * (bounds[i, 1] - bounds[i, 0]) + bounds[i, 0]
        return lhs_samples

    elif method == 1:
        dependency = np.diag(ps)
        latent = np.random.randn(p_samples, len(pm))
        return latent @ dependency + pm

    else:
        raise ValueError("Unsupported method. Choose 0 (LHS) or 1 (Gaussian).")


def data_gen(p_samples, x_in, u, t, noise_level=0.0, method=0):
    """Generate simulated data using different reactor types and kinetic parameters."""
    # Mean and standard deviation for parameter sampling
    pm = [16.7, 5.93, 0.28, 0.06, 0.16, 1.49]
    ps = [0.07, 0.60, 0.39, 0.28, 6.11, 3060]

    param_samples = generate_parameter_samples(p_samples, pm, ps, method)

    data = pd.DataFrame(columns=['x1', 'x2', 'x3', 'x4'])
    label_counter = 0

    for reactor_type in range(2):
        for kinetic_model in range(2):
            x_meas = np.zeros((p_samples, len(x_in)))

            for i in range(p_samples):
                theta = [
                    log(param_samples[i, 0]),
                    0.1 * param_samples[i, 1],
                    param_samples[i, 2],
                    param_samples[i, 3],
                    param_samples[i, 4],
                    param_samples[i, 5],
                ]
                x_out = model_solve(t, x_in, u, theta, reactor_type, kinetic_model)

                noise = np.random.multivariate_normal(
                    mean=np.zeros(4),
                    cov=np.diag(np.square(noise_level * np.array(x_out)))
                )
                x_out_noisy = np.clip(np.array(x_out) + noise, a_min=0, a_max=None)

                x_meas[i, :] = x_out_noisy

            df_temp = pd.DataFrame(x_meas, columns=['x1', 'x2', 'x3', 'x4'])
            df_temp['label'] = label_counter
            data = pd.concat([data, df_temp], ignore_index=True)
            label_counter += 1

    return data


def data_gen_2(p_samples, x_in, u, t, noise_level=0.0, method=0):
    """Extended data generator for multiple inputs."""
    pm = [16.7, 5.93, 0.27, 0.06, 0.16, 1.49]
    ps = [0.02, 0.2, 0.1, 0.28, 6.11, 3060]

    param_samples = generate_parameter_samples(p_samples, pm, ps, method)

    input_dim = x_in.shape[1] * x_in.shape[0]
    data = pd.DataFrame()
    label_counter = 0

    for reactor_type in range(2):
        for kinetic_model in range(2):
            x_meas = np.zeros((p_samples, input_dim))

            for i in range(p_samples):
                theta = [
                    log(param_samples[i, 0]),
                    0.1 * param_samples[i, 1],
                    param_samples[i, 2],
                    param_samples[i, 3],
                    param_samples[i, 4],
                    param_samples[i, 5],
                ]

                for k in range(x_in.shape[0]):
                    x_out = model_solve(t[k], x_in[k], u[k], theta, reactor_type, kinetic_model)
                    noise = np.random.multivariate_normal(
                        mean=np.zeros(4),
                        cov=np.diag(np.square(noise_level * np.array(x_out)))
                    )
                    x_out_noisy = np.clip(np.array(x_out) + noise, a_min=0, a_max=None)
                    x_meas[i, 4*k:4*(k+1)] = x_out_noisy

            df_temp = pd.DataFrame(x_meas)
            df_temp['label'] = label_counter
            data = pd.concat([data, df_temp], ignore_index=True)
            label_counter += 1

    return data