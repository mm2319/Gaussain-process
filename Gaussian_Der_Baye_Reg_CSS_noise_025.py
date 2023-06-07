from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pandas
import derivative
from lorenz import create_data_lorenz
from non_linear import create_data_nonlinear
from two_compartment import create_data_twocompart
from Derivative_Data_Lorenz import obtain_train_data_Lorenz
from Derivative_Data_NonLinear import obtain_train_data_NonLinear
from Derivative_Data_Two_Compart import obtain_train_data_Two_compart
from Bayesian_Regression_Disc_Spike_and_Slab import Bayesian_regression_disc_spike_slab
from Bayesian_Regression_Cont_Spike_and_Slab import Bayesian_regression_conti_spike_slab
from Bayesian_Regression_SS_Selection_2 import Bayesian_regression_SS_Selction
from Gaussian_process_der import GP, GP_derivative,rbf,rbf_fd,rbf_pd_2,rbf_pd_1,RBF_partial_diff_first,RBF_partial_diff_second
from sklearn.datasets import make_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from scipy.optimize import minimize
import autograd.numpy as np
from autograd import value_and_grad
from skopt.plots import plot_convergence
np.random.seed(0)
para_two_compart_1 = np.load('gp_025_para_tc_1.npy')
para_two_compart_2 = np.load('gp_025_para_tc_2.npy')
para_nonlinear_1  = np.load('gp_025_para_nl_1.npy')
para_nonlinear_2 = np.load('gp_025_para_nl_2.npy')
para_lorenz_1 =np.load('gp_025_para_lorenz_1.npy')
para_lorenz_2 =np.load('gp_025_para_lorenz_2.npy')
para_lorenz_3 =np.load('gp_025_para_lorenz_3.npy')
gp = GP(kernel=rbf,kernel_diff=rbf_pd_1)
T, Y_tc = create_data_twocompart(p=0.25)
Y_compart = []
y_pred_1 = gp.predict_mean(
              x_star=np.arange(0,10,0.1),  # set to test points
              X = np.array(T),     # set to observed x
              y = np.array(Y_tc[:,0]),       # set to observed y
              size=1,    # draw 100 posterior samples 
              theta=[para_two_compart_1[0],para_two_compart_1[1]],
              sigma=para_two_compart_1[2]
              )
y_pred_2 = gp.predict_mean(
              x_star=np.arange(0,10,0.1),  # set to test points
              X = np.array(T),     # set to observed x
              y = np.array(Y_tc[:,1]),       # set to observed y
              size=1,    # draw 100 posterior samples 
              theta=[para_two_compart_2[0],para_two_compart_2[1]],
              sigma=para_two_compart_2[2]
              )
Y_compart.append(y_pred_1)
Y_compart.append(y_pred_2)
Y_compart = np.array(Y_compart).T

T, Y_nl = create_data_nonlinear(p=0.25)
Y_nonlinear = []
y_pred_1 = gp.predict_mean(
              x_star=np.arange(0,10,0.1),  # set to test points
              X = np.array(T),     # set to observed x
              y = np.array(Y_nl[:,0]),       # set to observed y
              size=1,    # draw 100 posterior samples 
              theta=[para_nonlinear_1[0],para_nonlinear_1[1]],
              sigma=para_nonlinear_1[2]
              )
y_pred_2 = gp.predict_mean(
              x_star=np.arange(0,10,0.1),  # set to test points
              X = np.array(T),     # set to observed x
              y = np.array(Y_nl[:,1]),       # set to observed y
              size=1,    # draw 100 posterior samples 
              theta=[para_nonlinear_2[0],para_nonlinear_2[1]],
              sigma=para_nonlinear_2[2]
              )
Y_nonlinear.append(y_pred_1)
Y_nonlinear.append(y_pred_2)
Y_nonlinear = np.array(Y_nonlinear).T

T, Y_lr = create_data_lorenz(p=0.25)
Y_lorenz = []
y_pred_1 = gp.predict_mean(
              x_star=np.arange(0,10,0.1),  # set to test points
              X = np.array(T),     # set to observed x
              y = np.array(Y_lr[:,0]),       # set to observed y
              size=1,    # draw 100 posterior samples 
              theta=[para_lorenz_1[0],para_lorenz_1[1]],
              sigma=para_lorenz_1[2]
              )
y_pred_2 = gp.predict_mean(
              x_star=np.arange(0,10,0.1),  # set to test points
              X = np.array(T),     # set to observed x
              y = np.array(Y_lr[:,1]),       # set to observed y
              size=1,    # draw 100 posterior samples 
              theta=[para_lorenz_2[0],para_lorenz_2[1]],
              sigma=para_lorenz_2[2]
              )
y_pred_3 = gp.predict_mean(
              x_star=np.arange(0,10,0.1),  # set to test points
              X = np.array(T),     # set to observed x
              y = np.array(Y_lr[:,2]),       # set to observed y
              size=1,    # draw 100 posterior samples 
              theta=[para_lorenz_3[0],para_lorenz_3[1]],
              sigma=para_lorenz_3[2]
              )
Y_lorenz.append(y_pred_1)
Y_lorenz.append(y_pred_2)
Y_lorenz.append(y_pred_3)
Y_lorenz= np.array(Y_lorenz).T

print("$"*25)
print("for the continuous spike and slab prior")
print("$"*25)


toy_xp = np.arange(0,10,0.1)
gp = GP_derivative(kernel=rbf, kernel_diff=rbf_fd)
result_1 = gp.predict(
x_star=toy_xp,  # set to test points
X = np.array(T),     # set to observed x
y = np.array(Y_tc[:,0]),       # set to observed y
size=1,    # draw 100 posterior samples 
theta=[para_two_compart_1[0],para_two_compart_1[1]],
sigma=para_two_compart_1[2]
)
toy_xp = np.arange(0,10,0.1)
gp = GP_derivative(kernel=rbf, kernel_diff=rbf_fd)
result_2 = gp.predict(
x_star=toy_xp,  # set to test points
X = np.array(T),     # set to observed x
y = np.array(Y_tc[:,1]),       # set to observed y
size=1,    # draw 100 posterior samples 
theta=[para_two_compart_2[0],para_two_compart_2[1]],
sigma=para_two_compart_2[2]
)

x_1_train, y_1_train, x_2_train, y_2_train  = obtain_train_data_Two_compart( result_1, result_2, num_samples = 100, Y = Y_compart)

start_1,trace_1 = Bayesian_regression_conti_spike_slab(y_1_train,x_1_train,np.shape(x_1_train[0])[0])
start_2,trace_2 = Bayesian_regression_conti_spike_slab(y_2_train,x_2_train,np.shape(x_1_train[0])[0])
print("the value of z_1 in model_1 of two compartment model is",start_1['z_1'])
print("the value of beta_1 in model_1 of two compartment model is",start_1['beta_1'])
print("the value of z_1 in model_2 of two compartment model is",start_2['z_1'])
print("the value of beta_1 in model_2 of two compartment model is",start_2['beta_1'])
np.save('gp_BR_CSS_000_tc_1',start_1['beta_1'])
np.save('gp_BR_CSS_000_tc_2',start_2['beta_1'])

result_1 = gp.predict(
x_star=toy_xp,  # set to test points
X = np.array(T),     # set to observed x
y = np.array(Y_nl[:,0]),       # set to observed y
size=1,    # draw 100 posterior samples 
theta=[para_nonlinear_1[0],para_nonlinear_1[1]],
sigma=para_nonlinear_1[2]
)
result_2 = gp.predict(
x_star=toy_xp,  # set to test points
X = np.array(T),     # set to observed x
y = np.array(Y_nl[:,1]),       # set to observed y
size=1,    # draw 100 posterior samples 
theta=[para_nonlinear_2[0],para_nonlinear_2[1]],
sigma=para_nonlinear_2[2]
)

x_1_train, y_1_train, x_2_train, y_2_train  = obtain_train_data_NonLinear( result_1, result_2, num_samples = 100, Y = Y_nonlinear)

start_1,trace_1 = Bayesian_regression_conti_spike_slab(y_1_train,x_1_train,np.shape(x_1_train[0])[0])
start_2,trace_2 = Bayesian_regression_conti_spike_slab(y_2_train,x_2_train,np.shape(x_1_train[0])[0])

print("the value of z_1 in model_1 of nonlinear compartment model is",start_1['z_1'])
print("the value of beta_1 in model_1 of nonlinear compartment model is",start_1['beta_1'])
print("the value of z_1 in model_2 of nonlinear compartment model is",start_2['z_1'])
print("the value of beta_1 in model_2 of nonlinear compartment model is",start_2['beta_1'])
np.save('gp_BR_CSS_000_nl_1',start_1['beta_1'])
np.save('gp_BR_CSS_000_nl_2',start_2['beta_1'])

result_1 = gp.predict(
x_star=toy_xp,  # set to test points
X = np.array(T),     # set to observed x
y = np.array(Y_lr[:,0]),       # set to observed y
size=1,    # draw 100 posterior samples 
theta=[para_lorenz_1[0],para_lorenz_1[1]],
sigma=para_lorenz_1[2]
)
result_2 = gp.predict(
x_star=toy_xp,  # set to test points
X = np.array(T),     # set to observed x
y = np.array(Y_lr[:,1]),       # set to observed y
size=1,    # draw 100 posterior samples 
theta=[para_lorenz_2[0],para_lorenz_2[1]],
sigma=para_lorenz_2[2]
)
result_3 = gp.predict(
x_star=toy_xp,  # set to test points
X = np.array(T),     # set to observed x
y = np.array(Y_lr[:,2]),       # set to observed y
size=1,    # draw 100 posterior samples 
theta=[para_lorenz_3[0],para_lorenz_3[1]],
sigma=para_lorenz_3[2]
)

x_1_train, y_1_train, x_2_train, y_2_train, x_3_train, y_3_train = obtain_train_data_Lorenz( result_1, result_2, result_3, num_samples = 100, y = Y_lorenz)

start_1,trace_1 = Bayesian_regression_conti_spike_slab(y_1_train,x_1_train,np.shape(x_1_train[0])[0])
start_2,trace_2 = Bayesian_regression_conti_spike_slab(y_2_train,x_2_train,np.shape(x_1_train[0])[0])
start_3,trace_3 = Bayesian_regression_conti_spike_slab(y_3_train,x_3_train,np.shape(x_1_train[0])[0])
print("the value of z_1 in model_1 of lorenz model is",start_1['z_1'])
print("the value of beta_1 in model_1 of lorenz model is",start_1['beta_1'])
print("the value of z_1 in model_2 of lorenz model is",start_2['z_1'])
print("the value of beta_1 in model_2 of lorenz model is",start_2['beta_1'])
print("the value of z_1 in model_3 of lorenz model is",start_3['z_1'])
print("the value of beta_1 in model_3 of lorenz model is",start_3['beta_1'])
np.save('gp_BR_CSS_000_lr_1',start_1['beta_1'])
np.save('gp_BR_CSS_000_lr_2',start_2['beta_1'])
np.save('gp_BR_CSS_000_lr_3',start_3['beta_1'])