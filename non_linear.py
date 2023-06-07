import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import random 
def nonlinear_oscillator(y, t):
    """
    This function is the the ODE functions and, this function is used for scipy.solve_ivp function to synthesis data

    Input:
    t: initial time
    y: initial vector (T0,H0,E0)
    """
    alpha  = -0.1
    beta= 2
    gamma=-2
    delta=-0.1
    dydt = [ alpha * y[0]**3 + beta*y[1]**3,
              gamma*y[0]**3 + delta*y[1]**3]
    return dydt

def add_percent_noise(v, p):
    random.seed(0)
    np.random.seed(0)
    for i in range(len(v)):
        noise = np.random.normal(0., np.abs(p*v[i]))
        v[i] = v[i] + noise


def create_data_nonlinear(U0=np.array([2, 0]), ts = np.arange(0,10,0.01), p =0):
    Y = odeint(nonlinear_oscillator, U0, ts)
    T = np.arange(0,10,0.01)
    add_percent_noise(Y[:,0], p = p)
    add_percent_noise(Y[:,1], p = p)
    return T, Y