import jax
import jax.numpy as jnp
import numpy as np
import optax
from plotting_utils import Plotting_Utils
import matplotlib.pyplot as plt
import os

class MSE_Data_Loss:
    def __init__(self, target_trj, start_idx, temporal_density):
        self.target_trj = target_trj
        eval_indices = jnp.arange(start_idx, self.target_trj.shape[0], 1)
        n = int(eval_indices.shape[0] * temporal_density)

        random_indices = np.random.choice(eval_indices, size=n, replace=False)
        self.eval_indices = eval_indices[random_indices]
        self.mask = np.zeros(self.target_trj.shape[0])
        self.mask[self.eval_indices] = 1
        self.mask = jnp.array(self.mask)

    def g(self, u, u_target, i):
        dirac_delta = jnp.any(self.eval_indices == i).astype(int)
        return dirac_delta * jnp.mean((u - u_target)**2)
    
    def data_loss(self, pred_trj):
        residual = (pred_trj - self.target_trj)
        mean = jnp.mean(residual**2, axis=1)
        return jnp.mean(self.mask * mean)
    
def DA_plots(target_trj, DA_trj, loss_record, param_target_record,  DT, DOMAIN_SIZE, root):
    DA_fig = Plotting_Utils.plot_trj(DA_trj, DT, DOMAIN_SIZE)

    target_fig_path = os.path.join(root, "target_trj.png")
    DA_fig_path = os.path.join(root, "DA_trj.png")
    DA_fig.savefig(DA_fig_path)
    plt.close(DA_fig)

    side_by_side_fig_path = os.path.join(root, "trj_comparison.png")
    DA_sum_path = os.path.join(root, "DA_summary.png")

    fig1 = Plotting_Utils.plot_side_by_side(target_trj, DA_trj, DT, DOMAIN_SIZE)
    fig1.savefig(side_by_side_fig_path)
    plt.close(fig1)
    fig2 = Plotting_Utils.plot_DA_comparison(target_trj, DA_trj, DT, DOMAIN_SIZE, loss_record, param_target_record)
    fig2.savefig(DA_sum_path)
    plt.close(fig2)

def Newton_with_line_search(hess_func, value_and_grad_func, params, params_target, num_iterations, rho, beta, min_step_size):
    loss_record = []
    param_target_record = []
    alpha_init=1.0

    @jax.jit
    def get_newton_step(hess, grad):
        hess_inv = jnp.linalg.pinv(hess, hermitian=True, rcond=1e-8)
        # Compute the Newton step
        newton_step = -hess_inv @ grad
        return newton_step



    for i in range(num_iterations):
        # Compute the current loss (objective function value) and gradient
        current_loss, grad = value_and_grad_func(params)
        loss_record.append(current_loss)
        param_target_record.append(jnp.mean(jnp.abs(params - params_target)))
        print(f"Newton Iteration {i+1} | Loss: {current_loss}")
        
        # Compute the Hessian and its pseudoinverse
        hess = hess_func(params)

        # Compute the Newton step
        newton_step = get_newton_step(hess, grad)
        
        # Backtracking line search to find an appropriate step size
        alpha = alpha_init

        while True:
            # Compute the new parameters with the current step size
            new_params = params + alpha * newton_step
            
            # Evaluate the new loss at the new parameters
            new_loss, _ = value_and_grad_func(new_params)
            
            # Check if the Armijo condition is satisfied
            if new_loss <= current_loss + beta * alpha * jnp.dot(grad, newton_step):
                break
            else:
                # Reduce alpha by a factor of rho
                if alpha <= min_step_size:
                    return params, loss_record, param_target_record
                alpha *= rho
                
        
        params = new_params

    return params, loss_record, param_target_record

def hybrid_optimzation(value_and_grad_func, hess_func, params, params_target, optimizer_settings, print_loss=False):
    num_iterations = optimizer_settings["num_iterations"]
    patience = optimizer_settings["patience"]
    factor = optimizer_settings["factor"]
    min_lr = optimizer_settings["min_lr"]
    learning_rate = optimizer_settings["lr"]
    newton_increment = optimizer_settings["newton_increment"]
    tol = optimizer_settings["tol"]

    #line search params
    num_newton_iterations = optimizer_settings["num_newton_iterations"]
    rho = optimizer_settings["rho"]
    beta = optimizer_settings["beta"]
    min_newton_step_size = optimizer_settings["min_newton_step_size"]


    best_loss = float('inf')
    patience_counter = 0

    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    loss_record = []
    param_target_record = []

    for i in range(num_iterations):
        current_loss, grad = value_and_grad_func(params)
        loss_record.append(current_loss)
        norm = jnp.linalg.norm(grad)
        # Gradient clipping
        if norm >= 1:
            grad = grad / norm * 1
        
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        param_target_record.append(jnp.mean(jnp.abs(params - params_target)))
        
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            learning_rate = max(learning_rate * factor, min_lr)
            optimizer = optax.adam(learning_rate=learning_rate)
            opt_state = optimizer.init(params)
            patience_counter = 0

        if i % 50 == 0:
            if print_loss:
                print(f"Iteration {i}, Loss: {current_loss:.6f}, LR: {learning_rate:.6e}")
        
        if i!=0 and (i) % newton_increment == 0:
            params, loss_record_newton, param_target_record_newton = Newton_with_line_search(hess_func, value_and_grad_func, params, params_target,
                                                                                             num_newton_iterations, rho, beta, min_newton_step_size)
            loss_record += loss_record_newton
            param_target_record += param_target_record_newton
            if loss_record_newton[-1] <= tol:
                return params, loss_record, param_target_record
            
    
    return params, loss_record, param_target_record

def grad_descent(value_and_grad_func, params, params_target, optimizer_settings, print_loss=False):
    num_iterations = optimizer_settings["num_iterations"]
    patience = optimizer_settings["patience"]
    factor = optimizer_settings["factor"]
    min_lr = optimizer_settings["min_lr"]
    learning_rate = optimizer_settings["lr"]

    best_loss = float('inf')
    patience_counter = 0

    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    loss_record = np.zeros(num_iterations)
    param_target_record = np.zeros(num_iterations)

    for i in range(num_iterations):
        current_loss, grad = value_and_grad_func(params)
        loss_record[i] = current_loss
        norm = jnp.linalg.norm(grad)
        # Gradient clipping
        if norm >= 1:
            grad = grad / norm * 1
        
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        param_target_record[i] = jnp.mean(jnp.abs(params - params_target))
        
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            learning_rate = max(learning_rate * factor, min_lr)
            optimizer = optax.adam(learning_rate=learning_rate)
            opt_state = optimizer.init(params)
            patience_counter = 0

        if i % 100 == 0:
            if print_loss:
                print(f"Iteration {i}, Loss: {current_loss:.6f}, LR: {learning_rate:.6e}")
    
    return params, loss_record, param_target_record
