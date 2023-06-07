import derivative
from scipy.integrate import odeint
import numpy as np
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pandas

def conti_spike_slab(shape):
    mu_hat = 0
    sigma_hat = 10
    tau = 5
    lambda_hat = pm.Normal('lambda_hat', mu = mu_hat, sigma = sigma_hat, shape = shape)
    spike_raw = pm.Normal('spike_raw', mu = 0, sigma = 1, shape = shape)
    spike = pm.Deterministic('spike',tau*spike_raw*pm.invlogit(lambda_hat))
    return spike

def Bayesian_regression_conti_spike_slab(Y_1, X_1, size_fun_lib):
    basic_model = pm.Model()
    Y1 = np.array(Y_1)
    X_1 = np.array(X_1)
    with basic_model:
        sigma = pm.Gamma('sigma',1.,0.1,shape=1)
        z_1  = pm.Laplace('z_1', mu=0., b=1., shape=size_fun_lib)        
        pn_1 = conti_spike_slab(size_fun_lib)
        beta_1 = pm.Deterministic('beta_1', z_1*pn_1)
        mu_1 = pm.Deterministic(name="mu_1", var = pm.math.matrix_dot(X_1,beta_1))
        Y_obs_1 = pm.Normal('Y_obs_1', mu=mu_1, sigma = sigma, observed = Y1)
    with basic_model:
        trace_rh = pm.sample(1000, tune=6000, cores=1, random_seed=1, nuts={'target_accept':0.9})
    with basic_model:
        start = {}
        start['beta_1'] = trace_rh.posterior['beta_1'][:,-1:]
        start['mu_1'] = trace_rh.posterior['mu_1'][:,-1:]
        start['spike'] = trace_rh.posterior['spike'][:,-1:]
        start['z_1'] = trace_rh.posterior['z_1'][:,-1:]
    return start, trace_rh
