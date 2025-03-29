from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, HBox, VBox, Button
import numpy as np
from IPython.display import display, clear_output
import time

# distance between trajectories. this might be slight so we can use scaling if needed in order to see.
divergence = np.linalg.norm(solution_ivp.y - solution_odeint.T, axis=0)

# Performance throttle to reduce processing load
performance_throttle = 5  # cut the steps processed by this amount for performance purposes

# Create output widget for controlled updates
plot_output = widgets.Output()

# Debounce mechanism
last_update_time = time.time()
update_pending = False
current_plot = None

# Interactive divergence visualization
def create_plot(steps=888, throttle=performance_throttle):
    # Fixed parameters that work well
    highlight_steps = 1.0
    divergence_scaling = 1.0
    
    # Calculate stride based on throttle
    stride = max(1, throttle)
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # trajectory from solve_ivp (baseline)
    x, y, z = solution_ivp.y[0, :steps], solution_ivp.y[1, :steps], solution_ivp.y[2, :steps]

    # divergence width calculation
    widths = divergence[:steps] * divergence_scaling

    # Colormap: recent steps purple, older gray
    colors = ['#BBBBBB'] * steps  # gray for old steps
    highlight_length = int(inv_dt * highlight_steps)
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

    ax.set_xlim(-20, 20)
    ax.set_ylim(-30, 30)
    ax.set_zlim(0, 50)
    ax.set_xlabel('x', labelpad=10)
    ax.set_ylabel('y', labelpad=10)
    ax.set_zlabel('z', labelpad=10)
    ax.set_title("Lorenz Divergence Visualization\n"
                 "Distances between solve_ivp (RK45) (default) and odeint\n"
                 f"(Steps: {steps}, Throttle: {throttle}x)", fontsize=14)
    
    return fig

def update_plot(steps, throttle):
    global current_plot, last_update_time, update_pending
    
    # Clear previous output and create new plot
    with plot_output:
        clear_output(wait=True)
        current_plot = create_plot(steps, throttle)
        plt.show(current_plot)
    
    last_update_time = time.time()
    update_pending = False

def debounced_update(change):
    global update_pending, last_update_time
    
    # Get current values
    steps = steps_slider.value
    throttle = throttle_slider.value
    
    # If an update is already pending, do nothing
    if update_pending:
        return
    
    # If less than 200ms since last update, delay the update
    current_time = time.time()
    if current_time - last_update_time < 0.2:
        update_pending = True
        # Schedule update after a short delay
        timer = widgets.Button(description="hidden")
        timer.layout.visibility = 'hidden'
        display(timer)
        
        def delayed_update(b):
            update_plot(steps, throttle)
            timer.close()
        
        timer.on_click(delayed_update)
        timer.click()
    else:
        # Update immediately
        update_plot(steps, throttle)

# Create widgets
steps_slider = widgets.IntSlider(
    min=100, 
    max=num_steps, 
    step=100, 
    value=888, 
    description="Steps",
    continuous_update=False  # Only update when released
)

# Create play widget - this will control the steps slider
play_button = widgets.Play(
    min=100,
    max=num_steps,
    step=100,
    interval=50,  # milliseconds between each frame (increased for better performance)
    value=888,
    description="Press play",
    disabled=False
)

# Create throttle slider
throttle_slider = widgets.IntSlider(
    min=1,
    max=10,
    step=1,
    value=performance_throttle,
    description="Throttle",
    continuous_update=False
)

# Link the play widget to the slider
widgets.jslink((play_button, 'value'), (steps_slider, 'value'))

# Connect observers
steps_slider.observe(debounced_update, names='value')
throttle_slider.observe(debounced_update, names='value')

# Display controls and output
controls = HBox([play_button, steps_slider, throttle_slider])
display(controls)
display(plot_output)

# Initial plot
update_plot(steps_slider.value, throttle_slider.value)