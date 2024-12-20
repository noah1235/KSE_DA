import sys
import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd

jax.config.update("jax_enable_x64", True)

# Set paths for the module imports
module_path = os.path.abspath(os.path.join('C:/Users/havan/OneDrive/Desktop/Honors_Thesis/KSE_DA_Study/Source_Code'))
AE_path = os.path.abspath(os.path.join('C:/Users/havan/OneDrive/Desktop/Honors_Thesis/KSE_DA_Study/Source_Code'))
sys.path.append(AE_path)
sys.path.append(module_path)
from KS_Integrators import KS_RK4, KS_RK3, run_solver

#from DA_Utils import MSE_Data_Loss
from Source_Code.plotting_utils import Plotting_Utils
from Source_Code.DA_Utils import MSE_Data_Loss, DA_plots, hybrid_optimzation, grad_descent
from Source_Code.adjoint_DA import Adjoint_DA
from Source_Code.Auto_Diff_DA import Autodiff_DA
from Source_Code.Init_generator import generate_init_trjs
from Source_Code.KS_Integrators import KS_RK4, KS_RK3, run_solver



def write_dict_to_text(path, dict):
    with open(path, "w") as file:
        for key, value in dict.items():
            file.write(f"{key}: {value}\n")

def init_trj_cases():
    init_trj_cases = [{"domain_size": 22, "N_DOF": 128, "DT": .1, "T": 1000, "jax_seed": 1, "Solver_type": KS_RK4}]
    generate_init_trjs(init_trj_cases)

def save_results_to_excel(file_path, new_data):
    # Check if the file exists
    if not os.path.exists(file_path):
        print("File does not exist. Creating a new Excel file...")
        # Create a new Excel file with the new data
        new_data.to_excel(file_path, index=False, sheet_name="Sheet1")
        print("New file created successfully!")
    else:
        print("File exists. Loading data...")
        # Load the existing Excel file
        try:
            existing_data = pd.read_excel(file_path, sheet_name="Sheet1")
            print("Existing data loaded successfully.")
        except Exception as e:
            print("Error loading existing data:", e)
            existing_data = pd.DataFrame()  # If an error occurs, assume no data.

        # Append the new data to the existing data
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)

        # Write the combined data back to the Excel file
        combined_data.to_excel(file_path, index=False, sheet_name="Sheet1")
        print("New data appended successfully!")

def Run_DA_Cases():
    #DA cases to run
    file_path = "DA_cases_input.xlsx"
    #DT used for DA
    DT = 0.1
    #min time for initial condition sampling
    min_sampling_time = 400
    #DT used to generate init trj
    init_trj_DT = .1
    #path to trajectory
    local_init_trj_path = "initial_trjs\\domain_22\\N_DOF_128\\DT=0.1_T=1000_seed=1_solver=RK4.npy\\DT=0.1_T=1000_seed=1_solver=RK4.npy"

    init_trj = np.load(local_init_trj_path)
    N_DOF = init_trj.shape[1]
    selection_lb = int(min_sampling_time / init_trj_DT)
    selection_ub = init_trj.shape[0]


    df = pd.read_excel(file_path)

    root = f"DA_Cases_DT={DT}_N_DOF={N_DOF}"
    excel_results_path = os.path.join(root, "results.xlsx")
    os.makedirs(root, exist_ok=True)

    #settings for ADAM optimizer
    ADAM_settings = {
        "lr": .001,
        "num_iterations": 1000,
        "patience": 10,
        "factor": .7,
        "min_lr": 1e-7
        }
    #Settings for hybrid ADAM and newton optimizer
    Hybrid_settings = {
        "lr": .001,
        "num_iterations": 400,
        "patience": 10,
        "factor": .7,
        "min_lr": 1e-7,
        "newton_increment": 200,
        "tol": 1e-12,
        "rho": 0.5,
        "beta": 1e-4,
        "min_newton_step_size": 1e-1,
        "num_newton_iterations": 200
    }

    results_dict = {
        "DOMAIN_SIZE": [],
        "time_horizon": [],
        "standoff_percent": [],
        "temporal_density": [],
        "pert_mag": [],
        "ADAM_Autodiff_U0_MAE": [],
        "ADAM_Autodiff_trj_MAE": [],
        "Hybrid_Autodiff_U0_MAE": [],
        "Hybrid_Autodiff_trj_MAE": [],
        "ADAM_Adjoint_U0_MAE": [],
        "ADAM_Adjoint_trj_MAE": [],
        "Hybrid_Adjoint_U0_MAE": [],
        "Hybrid_Adjoint_trj_MAE": []
    }

    for index, row in df.iterrows():
        for _ in range(row["num_repeats"]):
            DOMAIN_SIZE = row["DOMAIN_SIZE"]

            time_horizon= row["time_horizon"]
            standoff_percent = row["standoff_percent"]
            temporal_density = row["temporal_density"]
            pert_mag = row["pert_mag"]

            Solver_type = KS_RK4
            loss_func = "MSE"

            results_dict["DOMAIN_SIZE"].append(DOMAIN_SIZE)
            results_dict["temporal_density"].append(temporal_density)
            results_dict["time_horizon"].append(time_horizon)
            results_dict["standoff_percent"].append(standoff_percent)
            results_dict["pert_mag"].append(pert_mag)

            DA_options = {
                "domain_size": DOMAIN_SIZE,
                "dt": DT,
                "time_horizon" : time_horizon,
                "standoff_percent": standoff_percent,
                "temporal_density": temporal_density,
                "pert_mag": pert_mag,
                "Solver": Solver_type.name
            }


            selection_index = np.random.randint(selection_lb, selection_ub)

            u_0 = init_trj[selection_index, :]
            pert = np.random.uniform(low=-1.0, high=1.0, size=N_DOF)
            u_0_guess = u_0 + pert * pert_mag

            time_steps = int(time_horizon / DT)
            start_index = int(standoff_percent * time_steps)

            solver = jax.jit(Solver_type(L=DOMAIN_SIZE, N=N_DOF, dt=DT))
            target_trj = run_solver(solver, u_0, time_steps)

            if loss_func == "MSE":
                loss_obj = MSE_Data_Loss(target_trj, start_index, temporal_density)

            domain_size_root = os.path.join(root, f"domain={DOMAIN_SIZE}")
            os.makedirs(domain_size_root, exist_ok=True)
            time_horizon_root = os.path.join(domain_size_root, f"time_horizon={time_horizon}")
            os.makedirs(time_horizon_root, exist_ok=True)
            temporal_density_root = os.path.join(time_horizon_root, f"temporal_density={temporal_density}")
            os.makedirs(temporal_density_root, exist_ok=True)
            standoff_percent_root = os.path.join(temporal_density_root, f"standoff_percent={standoff_percent}")
            os.makedirs(standoff_percent_root, exist_ok=True)
            loss_func_root = os.path.join(standoff_percent_root, f"{loss_func}")
            os.makedirs(loss_func_root, exist_ok=True)

            current_datetime = datetime.now()
            test_name = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
            test_root = os.path.join(loss_func_root, test_name)

            ADAM_root = os.path.join(test_root, f"ADAM_Optimization")
            os.makedirs(ADAM_root, exist_ok=True)
            
            hybrid_root = os.path.join(test_root, f"Hybrid_Optimization")
            os.makedirs(hybrid_root, exist_ok=True)
            
            ADAM_auto_diff_root = os.path.join(ADAM_root, f"Autodiff")
            os.makedirs(ADAM_auto_diff_root, exist_ok=True)

            Hybrid_auto_diff_root = os.path.join(hybrid_root, f"Autodiff")
            os.makedirs(Hybrid_auto_diff_root, exist_ok=True)
            
            ADAM_adjoint_root = os.path.join(ADAM_root, f"Adjoint")
            os.makedirs(ADAM_adjoint_root, exist_ok=True)

            hybrid_adjoint_root = os.path.join(hybrid_root, "Adjoint")
            os.makedirs(hybrid_adjoint_root, exist_ok=True)


            DA_options_txt_path = os.path.join(test_root, "DA_Options.txt")
            write_dict_to_text(DA_options_txt_path, DA_options)
            u_0_guess_path = os.path.join(test_root, "u_0_guess.npy")
            target_trj_path = os.path.join(test_root, "target_trj.npy")
            np.save(u_0_guess_path, u_0_guess)
            np.save(target_trj_path, target_trj)

            target_fig = Plotting_Utils.plot_trj(target_trj, DT, DOMAIN_SIZE)
            target_fig.savefig(os.path.join(test_root, "target_trj.png"))
            plt.close(target_fig)

            DA_mask_path = os.path.join(test_root, "DA_mask.npy")
            DA_mask_fig_path = os.path.join(test_root, "DA_mask_heat_map.png")
            DA_mask = loss_obj.mask
            DA_mask_fig = Plotting_Utils.plot_DA_mask(DA_mask, DT, DOMAIN_SIZE)
            np.save(DA_mask_path, DA_mask)
            DA_mask_fig.savefig(DA_mask_fig_path)
            plt.close(DA_mask_fig)
            

            ADAM_options_txt_path = os.path.join(ADAM_root, "ADAM_settings.txt")
            write_dict_to_text(ADAM_options_txt_path, ADAM_settings)
            
            hybrid_options_txt_path = os.path.join(hybrid_root, "Hybrid_settings.txt")
            write_dict_to_text(hybrid_options_txt_path, Hybrid_settings)

            def save_case(u_0_optimized, loss_record, param_target_record, root, key1, key2):
                DA_trj = run_solver(solver, u_0_optimized, time_steps)
                u0_MAE = float(jnp.mean(jnp.abs(u_0_optimized - u_0)))
                trj_MAE = float(jnp.mean(jnp.abs(DA_trj-target_trj)))
                results_dict[key1].append(u0_MAE)
                results_dict[key2].append(trj_MAE)
                DA_plots(target_trj, DA_trj, loss_record, param_target_record, DT, DOMAIN_SIZE, root)
                np.save(os.path.join(root, "DA_trj.npy"), DA_trj)
            
            adjoint_value_and_grad_func, adjoint_hess_func = Adjoint_DA(u_0, solver, time_steps, loss_obj)
            autodiff_value_and_grad_func, autodiff_hess_func = Autodiff_DA(solver, time_steps, loss_obj)


            #Autodiff Hybrid
            u_0_optimized, loss_record, param_target_record = hybrid_optimzation(autodiff_value_and_grad_func, autodiff_hess_func, u_0_guess, u_0, Hybrid_settings, print_loss=True)
            save_case(u_0_optimized, loss_record, param_target_record, Hybrid_auto_diff_root, "Hybrid_Autodiff_U0_MAE", "Hybrid_Autodiff_trj_MAE")

        
            #Autodiff ADAM
            u_0_optimized, loss_record, param_target_record = grad_descent(autodiff_value_and_grad_func, u_0_guess, u_0, Hybrid_settings, print_loss=True)
            save_case(u_0_optimized, loss_record, param_target_record, ADAM_auto_diff_root, "ADAM_Autodiff_U0_MAE", "ADAM_Autodiff_trj_MAE")

            #Adjoint Hybrid
            u_0_optimized, loss_record, param_target_record = hybrid_optimzation(adjoint_value_and_grad_func, adjoint_hess_func, u_0_guess, u_0, Hybrid_settings, print_loss=True)
            save_case(u_0_optimized, loss_record, param_target_record, hybrid_adjoint_root, "Hybrid_Adjoint_U0_MAE", "Hybrid_Adjoint_trj_MAE")

            #Adjoint ADAM
            u_0_optimized, loss_record, param_target_record = grad_descent(adjoint_value_and_grad_func, u_0_guess, u_0, Hybrid_settings, print_loss=True)
            save_case(u_0_optimized, loss_record, param_target_record, ADAM_adjoint_root, "ADAM_Adjoint_U0_MAE", "ADAM_Adjoint_trj_MAE")
            print(results_dict)
    print(results_dict)
    new_data_df = pd.DataFrame(results_dict)
    save_results_to_excel(excel_results_path, new_data_df)

#init_trj_cases()
Run_DA_Cases()
