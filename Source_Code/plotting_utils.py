import matplotlib.pyplot as plt
import numpy as np

class Plotting_Utils:
    @staticmethod
    def plot_DA_mask(data, DT, L):
        data_2d = data.reshape(1, -1)  # Reshape to 2D (1 row, many columns)

        # Create the heatmap figure
        fig, ax = plt.subplots(figsize=(18, 6))  # Use subplots to get figure and axes
        cax = ax.imshow(
            data_2d, cmap='coolwarm', aspect='auto',
            extent=(0, data.shape[0] * DT, -L / 2, L / 2),
            vmin=0,
            vmax=1,
        )

        # Customize the axes
        ax.set_yticks([])  # Remove y-axis ticks for simplicity
        ax.set_xlabel("Index")
        ax.set_title("DA Mask Heatmap")

        # Add a color bar
        fig.colorbar(cax, ax=ax, label="Value")

        # Return the figure object instead of showing it
        return fig


    @staticmethod
    def plot_trj(trj, DT, L, axes=None):
        n_dof = trj.shape[1]
        num_snapshots = trj.shape[0]
        u_0 = trj[0, :]  # Initial snapshot
        middle_snapshot = trj[int(num_snapshots * 0.5), :]  # Middle snapshot
        final_snapshot = trj[num_snapshots - 1, :]  # Final snapshot

        space_axis = np.linspace(-L / 2, L / 2, n_dof)

        # If no axes provided, create a new figure and axes
        if axes is None:
            fig, axes = plt.subplots(2, 1, figsize=(10, 10))

        # First subplot: Snapshots
        axes[0].plot(space_axis, u_0, label="Initial Snapshot")
        axes[0].plot(space_axis, middle_snapshot, label="Middle Snapshot")
        axes[0].plot(space_axis, final_snapshot, label="Final Snapshot")
        axes[0].set_title("Snapshots at Different Times")
        axes[0].set_xlabel("Space")
        axes[0].set_ylabel("Value")
        axes[0].legend(loc="best")
        axes[0].grid()

        # Second subplot: Space-Time plot
        im = axes[1].imshow(
            trj.T,
            cmap="RdBu",
            aspect="auto",
            origin="lower",
            extent=(0, trj.shape[0] * DT, -L / 2, L / 2),
            vmin=-2,
            vmax=2,
        )
        axes[1].set_title("Space-Time Plot")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Space")

        # Add colorbar
        fig = axes[1].get_figure()
        fig.colorbar(im, ax=axes[1], orientation="vertical", fraction=0.05, pad=0.05, label="Amplitude")

        return fig

    @staticmethod
    def plot_DA_comparison(target_trj, DA_trj, DT, L, loss_record, param_target_record):
        """
        Compare target trajectory and DA trajectory with snapshots and a single percent difference plot.

        Parameters:
            target_trj (ndarray): Target trajectory matrix (snapshots x space).
            DA_trj (ndarray): DA trajectory matrix (snapshots x space).
            DT (float): Time step.
            L (float): Length of the spatial domain.
        """
        n_dof = target_trj.shape[1]
        num_snapshots = target_trj.shape[0]

        # Extract snapshots
        u_0_target = target_trj[0, :]  
        middle_snapshot_target = target_trj[int(num_snapshots * 0.5), :]  
        final_snapshot_target = target_trj[num_snapshots - 1, :]  

        u_0_DA = DA_trj[0, :]  
        middle_snapshot_DA = DA_trj[int(num_snapshots * 0.5), :]  
        final_snapshot_DA = DA_trj[num_snapshots - 1, :]  

        space_axis = np.linspace(-L / 2, L / 2, n_dof)

        # Calculate percent difference
        percent_diff = np.abs(target_trj - DA_trj) / (np.abs(target_trj) + 1e-8) * 100

        # Create a figure with a custom gridspec layout
        fig = plt.figure(figsize=(18, 10))
        spec = fig.add_gridspec(2, 3)

        # Top row: Individual snapshots
        ax1 = fig.add_subplot(spec[0, 0])
        ax2 = fig.add_subplot(spec[0, 1])
        ax3 = fig.add_subplot(spec[0, 2])

        # Bottom row: Single large space-time percent difference plot
        ax4 = fig.add_subplot(spec[1, :2])
        ax5 = fig.add_subplot(spec[1, 2])

        # Plot snapshots
        ax1.plot(space_axis, u_0_target, label="Target")
        ax1.plot(space_axis, u_0_DA, label="DA")
        ax1.set_title("Initial Condition Comparison")
        ax1.set_xlabel("Space")
        ax1.set_ylabel("Value")
        ax1.legend(loc="best")
        ax1.grid()

        ax2.plot(space_axis, middle_snapshot_target, label="Target")
        ax2.plot(space_axis, middle_snapshot_DA, label="DA")
        ax2.set_title("Middle Snapshot Comparison")
        ax2.set_xlabel("Space")
        ax2.set_ylabel("Value")
        ax2.legend(loc="best")
        ax2.grid()

        ax3.plot(space_axis, final_snapshot_target, label="Target")
        ax3.plot(space_axis, final_snapshot_DA, label="DA")
        ax3.set_title("Final Snapshot Comparison")
        ax3.set_xlabel("Space")
        ax3.set_ylabel("Value")
        ax3.legend(loc="best")
        ax3.grid()

        # Bottom row: Single space-time percent difference plot
        im = ax4.imshow(percent_diff.T,
                        #cmap="RdBu",
                        aspect="auto",
                        origin="lower",
                        extent=(0, target_trj.shape[0] * DT, -L / 2, L / 2),
                        vmin=0,
                        vmax=100,)
        ax4.set_title("Space-Time Percent Difference")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Space")
        fig.colorbar(im, ax=ax4, orientation="vertical", fraction=0.05, pad=0.05, label="Percent Difference (%)")


        # Primary y-axis for "Data loss"
        ax5.set_title("Loss vs Iterations")
        ax5.set_xlabel("Iterations (Log Scale)")

        # First y-axis (left) for "Data loss"
        ax5.set_ylabel("Data Loss", color="b")
        ax5.plot(loss_record, 'b-', label="Data Loss")  # Blue solid line
        ax5.tick_params(axis='y', labelcolor="b")  # Left y-axis in blue

        param_target_record = np.array(param_target_record)
        # Secondary y-axis (right) for "Initial Condition MSE"
        ax5_right = ax5.twinx()  # Create second y-axis sharing the same x-axis
        ax5_right.set_ylabel("Initial Condition MAE", color="r")  # Right y-axis label
        ax5_right.plot(param_target_record, 'r--', label="Initial Condition MAE")  # Red dashed line
        ax5_right.tick_params(axis='y', labelcolor="r")  # Right y-axis in red

        # Set logarithmic scale for both axes
        ax5.set_xscale('log')  # Logarithmic x-axis
        ax5.set_yscale('log')  # Logarithmic scale for left y-axis
        ax5_right.set_yscale('log')  # Logarithmic scale for right y-axis

        # Add combined legend
        lines_1, labels_1 = ax5.get_legend_handles_labels()
        lines_2, labels_2 = ax5_right.get_legend_handles_labels()
        ax5.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

        fig.tight_layout()

        return fig

    @staticmethod
    def plot_side_by_side(target_trj, DA_trj, DT, L):
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 10))

        # First trajectory: Use first two columns
        Plotting_Utils.plot_trj(target_trj, DT, L, axes=[axes[0, 0], axes[1, 0]])
        axes[0, 0].set_title("Target Trajectory - Snapshots")
        axes[1, 0].set_title("Target Trajectory - Space-Time")

        # Second trajectory: Use next two columns
        Plotting_Utils.plot_trj(DA_trj, DT, L, axes=[axes[0, 1], axes[1, 1]])
        axes[0, 1].set_title("DA Trajectory - Snapshots")
        axes[1, 1].set_title("DA Trajectory - Space-Time")

        # Adjust layout
        fig.tight_layout()
        return fig

    

    
