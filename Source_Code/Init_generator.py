import os
import numpy as np
import jax
import jax.numpy as jnp
from KS_Integrators import run_solver
from Source_Code.plotting_utils import Plotting_Utils

def generate_init_trjs(trj_cases):
    # Move up one folder and define the path for "initial_trjs"
    #base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))  # Move up one level
    initial_trjs_dir = "initial_trjs"

    # Create "initial_trjs" folder if it doesn't exist
    os.makedirs(initial_trjs_dir, exist_ok=True)

    for trj_case in trj_cases:
        # Extract parameters
        DOMAIN_SIZE = trj_case["domain_size"]
        N_DOF = trj_case["N_DOF"]
        DT = trj_case["DT"]
        T = trj_case["T"]
        jax_seed = trj_case["jax_seed"]
        Solver_type = trj_case["Solver_type"]

        # Generate trajectory
        key = jax.random.PRNGKey(jax_seed)
        u_0 = jax.random.uniform(key, shape=(N_DOF,), minval=-1.0, maxval=1.0)
        time_steps = int(T / DT)
        solver = jax.jit(Solver_type(L=DOMAIN_SIZE, N=N_DOF, dt=DT))
        trj = run_solver(solver, u_0, time_steps)

        fig = Plotting_Utils.plot_trj(trj, DT, DOMAIN_SIZE)
        
        # Create a subfolder based on domain size
        domain_dir = os.path.join(initial_trjs_dir, f"domain_{DOMAIN_SIZE}")
        N_DOF_dir = os.path.join(domain_dir, f"N_DOF_{N_DOF}")
        os.makedirs(domain_dir, exist_ok=True)
        os.makedirs(N_DOF_dir, exist_ok=True)

        # Define the filename based on the case parameters
        file_name = f"DT={DT}_T={T}_seed={jax_seed}_solver={Solver_type.name}.npy"
        folder_path = os.path.join(N_DOF_dir, file_name)
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, file_name)
        fig_path = os.path.join(folder_path, "plot.png")

        # Save the trajectory as a NumPy array
        np.save(file_path, np.array(trj))  # Save as a NumPy file
        fig.savefig(fig_path)

        print(f"Saved trajectory: {file_path} with shape {trj.shape}")
