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
def optim_hyperparams(init_params, data_X, data_y, gp, method="L-BFGS-B", maxiter=500):
  """
  Find the best kernel hyper-parameters that maximise the log marginal
  likelihood.
  """
  # define negative log marginal likelihood as objective
  # input is unpacked to theta and sigma
  gp = GP(kernel=rbf,kernel_diff=rbf_pd_1)
  objective = lambda params: gp.loglikelihood(
                x_star=np.arange(0,10,0.01),  # set to test points
                X = data_X,     # set to observed x
                y = data_y,       # set to observed y
                size=1,    # draw 100 posterior samples 
                theta=params[:-1],
                sigma=params[-1]
              )

  optim_res = minimize(
      fun=value_and_grad(objective),
      jac=True,
      x0=init_params, 
      method=method,
      bounds = [[1e-10, 1.e3], [1e-10, 1.e3], [1e-10, 1.e3]],
      options={"return_all": True, "maxiter": maxiter}
  )
  return optim_res

def optim_hyperparams_multiple_runs(init_params_list, data_X, data_y, gp, maxiter=500):
  """
  Run hyper-parameter search with multiple starting points.
  """
  optim_res_list = []
  log_lik_history_list = []
  nrep = len(init_params_list)

  for i in range(nrep):
    init_params = init_params_list[i]

    # find best params
    optim_res = optim_hyperparams(
        init_params, data_X, data_y, gp, method="BFGS", maxiter=maxiter,
    )
    # log_lik_history = show_optim_history(optim_res.allvecs, gp, data_X, data_y)
    params = optim_res.x
    # compute log marginal lkhd
    log_lik = gp.loglikelihood(
                x_star=np.arange(0,10,0.01),  # set to test points
                X = data_X,     # set to observed x
                y = data_y,       # set to observed y
                size=1,    # draw 100 posterior samples 
                theta=params[:-1],
                sigma=params[-1]
              )

    # store results
    optim_res_list.append(optim_res)
    log_lik_history_list.append(log_lik)

  return optim_res_list, log_lik_history_list 
def grid_search_initial():
  init_params_list = []
  init_params = [1.e-3,1.e-2,1.e-1,1,10,100]
  for i in init_params:
    for j in init_params:
      for k in init_params:
        init_params_list.append([i,j,k])
  return np.array(init_params_list)
init_params_list = grid_search_initial()
gp = GP(kernel=rbf,kernel_diff=rbf_pd_1)
T, Y_lr = create_data_lorenz(p=0.0)
optim_res_list, log_lik_history_list  = optim_hyperparams_multiple_runs(
    init_params_list, T, Y_lr[:,0], gp)
best_idx = np.argmin(log_lik_history_list)
best_optim_res = optim_res_list[best_idx]
best_log_lik = log_lik_history_list[best_idx]
para_lorenz_1 = best_optim_res
np.save('gp_000_para_lr_1',para_lorenz_1.x)