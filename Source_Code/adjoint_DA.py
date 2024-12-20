import jax
import jax.numpy as jnp
from KS_Integrators import run_solver
from DA_Utils import MSE_Data_Loss, grad_descent, hybrid_optimzation
from Source_Code.plotting_utils import Plotting_Utils
import matplotlib.pyplot as plt
import numpy as np
from jax import lax

class Discrete_Adjoint_Grad_Hess():
    def __init__(self, dg__du, dstepper__du, hess_g, stepper_hess, N_DOF):
        self.dg__du= dg__du
        self.dstepper__du = dstepper__du
        self.hess_g = hess_g
        self.stepper_hess = stepper_hess
        self.N_DOF = N_DOF

    

    def __call__(self, lam_n_1, dlam_n_1__du_0, u_n, u_n_target, du_n__du_0, i):
        lam_n = self.dstepper__du(u_n).T @ lam_n_1 + self.dg__du(u_n, u_n_target, i).reshape((-1, 1))

        loss_hess_term = self.hess_g(u_n, u_n_target, i) @ du_n__du_0
        hess_dot = (lam_n_1.T @ self.stepper_hess(u_n)).reshape((self.N_DOF, self.N_DOF))
        
        dlam_n__du_0 = self.dstepper__du(u_n).T @ dlam_n_1__du_0 + hess_dot @ du_n__du_0 + loss_hess_term

        return lam_n, dlam_n__du_0


class Discrete_Adjoint_Grad_Stepper():
    def __init__(self, dg__du, dstepper__du):
        self.dg__du = dg__du
        self.dstepper__du = dstepper__du

    def __call__(self, lam_n_1, u_n, u_n_target, i):
        lam_n = self.dstepper__du(u_n).T @ lam_n_1 + self.dg__du(u_n, u_n_target, i).reshape((-1, 1))
        return lam_n


def get_trj_sens(trj, dstepper__du, time_steps, N_DOF):
    S_0 = jnp.eye(N_DOF)
    S = S_0

    def step_fn(S_current, i):
        u = trj[i, :]
        S = dstepper__du(u) @ S_current 

        return S, S
    indices = jnp.arange(0, time_steps, 1)
    S_trj = jax.lax.scan(step_fn, S_0, xs=indices)[1]
    S_trj = jnp.concatenate([jnp.expand_dims(S_0, 0), S_trj], axis=0)
    return S_trj

def get_trj_2sens(trj, dstepper__du, time_steps, N_DOF):
    S_0 = jnp.eye(N_DOF)
    S = S_0
    sens_trj = jnp.zeros((time_steps + 1, N_DOF, N_DOF))
    sens_trj = sens_trj.at[0].set(S_0)
    for i in range(time_steps):
        u = trj[i, :]
        S = dstepper__du(u) @ S
        sens_trj = sens_trj.at[i + 1].set(S)

    return sens_trj


def Adjoint_DA(u_0, solver, time_steps, loss_obj):
    N_DOF = u_0.shape[0]
    dg__du = jax.jit(jax.grad(loss_obj.g))
    hess_g = jax.jit(jax.hessian(loss_obj.g))

    dstepper__du = jax.jit(jax.jacobian(solver))
    stepper_hess = jax.jit(jax.hessian(solver))

    adj_stepper = jax.jit(Discrete_Adjoint_Grad_Stepper(dg__du, dstepper__du))
    
    adj_hess_stepper = jax.jit(Discrete_Adjoint_Grad_Hess(dg__du, dstepper__du, hess_g, stepper_hess, N_DOF))

    target_trj = loss_obj.target_trj

    @jax.jit
    def data_loss_func(pred_trj):
        return loss_obj.data_loss(pred_trj)

    @jax.jit
    def hess_func(u_0):
        trj = run_solver(solver, u_0, time_steps)
        sens_trj = get_trj_sens(trj, dstepper__du, time_steps, N_DOF)

        u_N = trj[-1, :]
        u_N_target = target_trj[-1, :]
        S_n = sens_trj[-1, :]
        lam_N = dg__du(u_N, u_N_target, time_steps).reshape((-1, 1))

        dlam_N__du_0 = hess_g(u_N, u_N_target, time_steps) @ S_n

        def scan_fn(carry, i):
            lam_n_1, dlam_n_1__du_0 = carry
            u_n = trj[i, :]
            u_n_target = target_trj[i, :]
            du_n__du_0 = sens_trj[i, :]

            lam_n_next, dlam_n__du_0 = adj_hess_stepper(lam_n_1, dlam_n_1__du_0, u_n, u_n_target, du_n__du_0, i)


            return (lam_n_next, dlam_n__du_0), None
        

        indices = jnp.arange(time_steps - 1, -1, -1)

        lam_0, dlam_0__du_0 = lax.scan(scan_fn, (lam_N, dlam_N__du_0), xs=indices)[0]

        return dlam_0__du_0

    @jax.jit
    def value_and_grad_func(u_0):
        trj = run_solver(solver, u_0, time_steps)
        
        u_N = trj[-1, :]
        u_N_target = target_trj[-1, :]
        lam_N = adj_stepper.dg__du(u_N, u_N_target, time_steps).reshape((-1, 1))

        def scan_fn(lam_n_1, i):
            u_n = trj[i, :]
            u_n_target = target_trj[i, :]

            lam_n_next = adj_stepper(lam_n_1, u_n, u_n_target, i)

            return lam_n_next, 0
        

        indices = jnp.arange(time_steps - 1, -1, -1)

        lam_0 = lax.scan(scan_fn, lam_N, xs=indices)[0]
        data_loss = data_loss_func(trj)

        return data_loss, lam_0.reshape(-1)

    def grad(u_0):
        _, grad = value_and_grad_func(u_0)
        return grad
    #hess = jax.jacobian(grad)

    return value_and_grad_func, hess_func
