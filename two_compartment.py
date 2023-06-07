import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import random

def two_compartment(y, t):
    """
    This function is the the ODE functions and, this function is used for scipy.solve_ivp function to synthesis data

    Input:
    t: initial time
    y: initial vector (T0,H0,E0)
    """
    F = 2.
    w = 3.
    gamma = 0.1
    N = y[0]
    K = y[1]
    dydt = [F*N*(1-(N/K)),  w*K - gamma*(N**(2/3))*K]
    
    return dydt

def add_percent_noise(v, p):
    random.seed(0)
    np.random.seed(0)
    for i in range(len(v)):
        noise = np.random.normal(0., np.abs(p*v[i]))
        v[i] = v[i] + noise
        if v[i] <= 0:
            v[i] = 1.e-5

def create_data_twocompart(U0=np.array([50, 800]), ts = np.arange(0,10,0.01), p = 0):
    Y = odeint(two_compartment, U0, ts)
    T = np.arange(0,10,0.01)
    add_percent_noise(Y[:,0], p = p)
    add_percent_noise(Y[:,1], p = p)
    return T, Y