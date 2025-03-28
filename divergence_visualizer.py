"""
Divergence Visualizer module for trajectory comparison and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import HBox, VBox
from IPython.display import display, clear_output
import time

class DivergenceVisualizer:
    """
    A class to visualize and animate divergence between trajectory solutions.
    """
    
    def __init__(self, solution_primary, solution_secondary=None, divergence=None, 
                 time_step=None, max_steps=None, default_steps=888):
        """
        Initialize the divergence visualizer.
        
        Parameters:
        -----------
        solution_primary : ndarray or dict-like
            Primary solution trajectory (e.g. from solve_ivp)
        solution_secondary : ndarray, optional
            Secondary solution trajectory (e.g. from odeint)
        divergence : ndarray, optional
            Pre-calculated divergence between trajectories. If None, will be calculated.
        time_step : float, optional
            Time step between solution points (for highlight calculation)
        max_steps : int, optional
            Maximum number of steps in the trajectory
        default_steps : int, optional
            Default number of steps to display
        """
        # Store solutions
        self.solution_primary = solution_primary
        
        # Extract y values based on solution type
        if hasattr(solution_primary, 'y'):
            # solve_ivp style
            self.y_primary = solution_primary.y
        else:
            # Assume array-like
            self.y_primary = solution_primary
            
        # Get dimensions
        self.dim = self.y_primary.shape[0]
        
        # Get max steps
        if max_steps is None:
            self.max_steps = self.y_primary.shape[1]
        else:
            self.max_steps = max_steps
            
        # Process secondary solution if provided
        if solution_secondary is not None:
            if hasattr(solution_secondary, 'y'):
                self.y_secondary = solution_secondary.y
            else:
                self.y_secondary = solution_secondary
                
            # Check if transpose needed
            if self.y_secondary.shape[0] != self.dim:
                self.y_secondary = self.y_secondary.T
        
        # Calculate or store divergence
        if divergence is not None:
            self.divergence = divergence
        elif solution_secondary is not None:
            self.divergence = np.linalg.norm(self.y_primary - self.y_secondary, axis=0)
        else:
            self.divergence = np.ones(self.max_steps) * 0.1  # Default
            
        # Store time step
        self.time_step = time_step if time_step is not None else 0.01
        self.inv_dt = 1.0 / self.time_step if time_step else 100
            
        # Visualization parameters
        self.default_steps = min(default_steps, self.max_steps)
        self.performance_throttle = 2
        
        # Debounce mechanism
        self.last_update_time = time.time()
        self.update_pending = False
        self.current_plot = None
        
        # Plot output widget
        self.plot_output = widgets.Output()
    
    def create_plot(self, steps, throttle):
        """
        Create the divergence visualization plot.
        
        Parameters:
        -----------
        steps : int
            Number of steps to display
        throttle : int
            Performance throttle factor
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure
        """
        # Fixed parameters
        highlight_steps = 1.0
        divergence_scaling = 1.0
        
        # Calculate stride based on throttle
        stride = max(1, throttle)
        
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Get coordinates
        x, y, z = self.y_primary[0, :steps], self.y_primary[1, :steps], self.y_primary[2, :steps]

        # Divergence width calculation
        widths = self.divergence[:steps] * divergence_scaling

        # Colormap: recent steps purple, older gray
        colors = ['#BBBBBB'] * steps  # gray for old steps
        highlight_length = int(self.inv_dt * highlight_steps)
        colors[-highlight_length:] = ['#800080'] * highlight_length  # purple for recent steps

        # Plot with proper throttling
        i_values = list(range(0, steps, stride))
        if steps-1 not in i_values:
            i_values.append(steps-1)  # Make sure we include the last point
            
        for i in range(len(i_values)-1):
            idx1, idx2 = i_values[i], i_values[i+1]
            ax.plot(x[idx1:idx2+1], y[idx1:idx2+1], z[idx1:idx2+1],
                    linewidth=max(widths[idx2], 0.1),
                    color=colors[idx2])

        # Set axis limits based on data range
        range_buffer = 0.1  # Add 10% buffer around data
        x_range = np.ptp(x)
        y_range = np.ptp(y)
        z_range = np.ptp(z)
        
        x_mid = (np.max(x) + np.min(x)) / 2
        y_mid = (np.max(y) + np.min(y)) / 2
        z_mid = (np.max(z) + np.min(z)) / 2
        
        ax.set_xlim(x_mid - x_range/2 * (1+range_buffer), x_mid + x_range/2 * (1+range_buffer))
        ax.set_ylim(y_mid - y_range/2 * (1+range_buffer), y_mid + y_range/2 * (1+range_buffer))
        ax.set_zlim(z_mid - z_range/2 * (1+range_buffer), z_mid + z_range/2 * (1+range_buffer))
        
        ax.set_xlabel('x', labelpad=10)
        ax.set_ylabel('y', labelpad=10)
        ax.set_zlabel('z', labelpad=10)
        ax.set_title("Trajectory Divergence Visualization\n"
                    f"(Steps: {steps}, Throttle: {throttle}x)", fontsize=14)
        
        return fig
    
    def update_plot(self, steps, throttle):
        """Update the plot with new parameters."""
        # Clear previous output and create new plot
        with self.plot_output:
            clear_output(wait=True)
            self.current_plot = self.create_plot(steps, throttle)
            plt.show(self.current_plot)
        
        self.last_update_time = time.time()
        self.update_pending = False

    def debounced_update(self, change):
        """Handle updates with debouncing to prevent excessive redrawing."""
        # Get current values
        steps = self.steps_slider.value
        throttle = self.throttle_slider.value
        
        # If an update is already pending, do nothing
        if self.update_pending:
            return
        
        # If less than 200ms since last update, delay the update
        current_time = time.time()
        if current_time - self.last_update_time < 0.2:
            self.update_pending = True
            # Schedule update after a short delay
            timer = widgets.Button(description="hidden")
            timer.layout.visibility = 'hidden'
            display(timer)
            
            def delayed_update(b):
                self.update_plot(steps, throttle)
                timer.close()
            
            timer.on_click(delayed_update)
            timer.click()
        else:
            # Update immediately
            self.update_plot(steps, throttle)
    
    def create_widgets(self):
        """Create the interactive widgets."""
        # Create slider for steps
        self.steps_slider = widgets.IntSlider(
            min=100, 
            max=self.max_steps, 
            step=100, 
            value=self.default_steps, 
            description="Steps",
            continuous_update=False
        )

        # Create play widget
        self.play_button = widgets.Play(
            min=100,
            max=self.max_steps,
            step=100,
            interval=1000,  # milliseconds between each frame
            value=self.default_steps,
            description="Press play",
            disabled=False
        )

        # Create throttle slider
        self.throttle_slider = widgets.IntSlider(
            min=1,
            max=5,
            step=1,
            value=self.performance_throttle,
            description="Throttle",
            continuous_update=False
        )

        # Link the play widget to the slider
        widgets.jslink((self.play_button, 'value'), (self.steps_slider, 'value'))

        # Connect observers
        self.steps_slider.observe(self.debounced_update, names='value')
        self.throttle_slider.observe(self.debounced_update, names='value')
    
    def display(self):
        """Display the interactive visualization."""
        # Create widgets if not already created
        if not hasattr(self, 'steps_slider'):
            self.create_widgets()
            
        # Display controls and output
        controls = HBox([self.play_button, self.steps_slider, self.throttle_slider])
        display(controls)
        display(self.plot_output)
        
        # Initial plot
        self.update_plot(self.steps_slider.value, self.throttle_slider.value)

# Helper function for easy creation
def visualize_divergence(solution_primary, solution_secondary=None, divergence=None, 
                         time_step=None, max_steps=None, default_steps=888):
    """
    Create and display an interactive divergence visualization.
    
    Parameters:
    -----------
    solution_primary : ndarray or dict-like
        Primary solution trajectory (e.g. from solve_ivp)
    solution_secondary : ndarray, optional
        Secondary solution trajectory (e.g. from odeint)
    divergence : ndarray, optional
        Pre-calculated divergence between trajectories. If None, will be calculated.
    time_step : float, optional
        Time step between solution points (for highlight calculation)
    max_steps : int, optional
        Maximum number of steps in the trajectory
    default_steps : int, optional
        Default number of steps to display
    """
    visualizer = DivergenceVisualizer(
        solution_primary, solution_secondary, divergence,
        time_step, max_steps, default_steps
    )
    visualizer.display()
    return visualizer