import jax
import jax.numpy as jnp
from KS_Integrators import run_solver
from DA_Utils import MSE_Data_Loss, grad_descent, Newton_with_line_search, hybrid_optimzation
from Source_Code.plotting_utils import Plotting_Utils
import matplotlib.pyplot as plt
import numpy as np


def Autodiff_DA(solver, time_steps, loss_obj):
    
    @jax.jit
    def objective(u_0_guess):
        pred_trj = run_solver(solver, u_0_guess, time_steps)
        return loss_obj.data_loss(pred_trj)
    
    value_and_grad_func = jax.jit(jax.value_and_grad(objective))
    hess_func = jax.jit(jax.hessian(objective))
    
    return value_and_grad_func, hess_func