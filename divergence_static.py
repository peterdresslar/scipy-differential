import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_ode_solves_comparison_static(
    solution_a: np.ndarray,
    solution_b: np.ndarray, 
    inv_dt: float,           # we do not assume that the legend_dict contains inv_dt (it's cosmetic)
    tolerance: float, 
    title_prefix: str,
    legend_dict: dict
):
    """
    Plots a static 2x2 figure comparing two solutions of a 3D ODE system.
    
    Parameters
    ----------
    solution_a : np.ndarray, shape (n, 3)
        The converted/normalized solution from solver A.
    solution_b : np.ndarray, shape (n, 3)
        The converted/normalized solution from solver B.
    inv_dt : float
        The inverse timestep (points per unit time) used when building these solutions.
    tolerance : float, optional
        A threshold for highlighting "convergence" points (where the difference is <= this value).
    title_prefix : str, optional
        A string prefix to apply to the overall figure title or subplots.
    legend_dict : dict, optional
        A dictionary of legend items to add to the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axs : np.ndarray
        The array of subplots (2D array of Axes) for further tweaking if desired.
    """
    # Validate input shape
    if solution_a.shape != solution_b.shape:
        raise ValueError("Solutions must have the same shape, e.g. (n, 3).")
    
    # Number of time steps
    n_steps = solution_a.shape[0]
    
    # recreate the time array from the inverse timestep
    # we do not assume that the legend_dict contains inv_dt (it could be changed)
    time_array = np.arange(n_steps) / inv_dt

    # Calculate absolute difference (vector norm) at each point
    diff = solution_a - solution_b
    # there are other normalizers but we use a basic one
    abs_diff = np.linalg.norm(diff, axis=1)
    
    # identify indices where difference is below tolerance
    # for chaotic systems, this is mostly a fun little check---not much meaning or periodicity in lorenz for instance
    below_tol_indices = np.where(abs_diff <= tolerance)[0]

    # Create a figure with four subplots (2x2)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': None})
    
    # -------------------------------------------------------------
    # Subplot 1: 3D comparison of the two trajectories
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax3d.plot(solution_a[:, 0], solution_a[:, 1], solution_a[:, 2],
              color='red', label='Solver A')
    ax3d.plot(solution_b[:, 0], solution_b[:, 1], solution_b[:, 2],
              color='blue', label='Solver B')

    # For clarity, we can use the coordinates from solution_a:
    ax3d.scatter(
        solution_a[below_tol_indices, 0],
        solution_a[below_tol_indices, 1],
        solution_a[below_tol_indices, 2],
        color='green', s=20, alpha=0.7,
        label=f"Within tolerance={tolerance}"
    )
    
    ax3d.set_title(f"{title_prefix} (3D Trajectories)")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.legend()
    
    # -------------------------------------------------------------
    # Subplot 2: Absolute norm difference vs. time
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(time_array, abs_diff, color='black', label='|| A - B ||')
    
    # Mark on the time axis where difference is below tolerance
    if below_tol_indices.size > 0:
        ax2.scatter(
            time_array[below_tol_indices],
            abs_diff[below_tol_indices],
            color='green',
            s=20, alpha=0.7
        )
    
    ax2.set_title(f"{title_prefix} (Norm Difference)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("||Solution A - Solution B||")
    ax2.legend()
    
    # -------------------------------------------------------------
    # Subplot 3: Coordinate-by-coordinate difference
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(time_array, diff[:, 0], label="Δx", color='r')
    ax3.plot(time_array, diff[:, 1], label="Δy", color='g')
    ax3.plot(time_array, diff[:, 2], label="Δz", color='b')
    
    # sub-threshold highlights might be interesting in some other system, but not terribly meaningful in lorenz
    # for i, label, c in zip(range(3), ["Δx", "Δy", "Δz"], ['r', 'g', 'b']):
    #     sub_diff = diff[:, i]
    #     # optionally add scatter points at below_tol_indices if needed
    #     pass  # just skip for a cleaner line plot

    ax3.set_title(f"{title_prefix} (Component Differences)")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Difference")
    ax3.legend()
    
    # -------------------------------------------------------------
    # Subplot 4: Facts
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    ax4.set_title("Summary")

    # Example summary stats:
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    # add in newlines to make legend_dict more readable
    legend_info = ""
    for key, value in legend_dict.items():
        legend_info += f"{key}: {value}\n"
    text_str = (
        f"Equation Inputs: {legend_info}\n"
        f"Max Norm Diff: {max_diff:.4f}\n"
        f"Mean Norm Diff: {mean_diff:.4f}\n"
        f"Indices below tolerance={tolerance}: {below_tol_indices.size} / {n_steps}"
    )
    ax4.text(0.1, 0.1, text_str, fontsize=12)

    # -------------------------------------------------------------
    fig.suptitle(f"{title_prefix} — Single Frame Comparison", fontsize=16)
    fig.tight_layout()

    return fig, axs