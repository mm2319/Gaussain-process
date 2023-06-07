import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import random
def Lorenz_system(y, t):
    sigma = 10
    beta = 8/3
    ro = 28
    dydt = [sigma*(y[1]-y[0]),
            y[0]*(ro-y[2])-y[1],
            y[0]*y[1]-beta*y[2]]
    return dydt

def add_percent_noise(v, p):
    random.seed(0)
    np.random.seed(0)
    for i in range(len(v)):
        noise = np.random.normal(0., np.abs(p*v[i]))
        v[i] = v[i] + noise



def create_data_lorenz(U0=np.array([-8., 7. ,27.]), ts = np.arange(0,10,0.01), p=0):
    Y = odeint(Lorenz_system, U0, ts)
    T = np.arange(0,10,0.01)
    add_percent_noise(Y[:,0], p = p)
    add_percent_noise(Y[:,1], p = p)
    add_percent_noise(Y[:,2], p = p)
    return T, Y