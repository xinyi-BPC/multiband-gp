import matplotlib.pyplot as plt
import numpy as np

def plot_gp_fit(gp, data, heldout_data=None, n_grid=300):
    """
    Plots the Gaussian Process fit along with the original data points.
    
    Args:
        gp: The fitted GaussianProcessRegressor model.
        data: A dictionary containing training 't', 'y', and 'yerr' arrays.
        heldout_data: Optional held-out observations to overlay.
        n_grid: Number of points in the grid for prediction.
    """
    t = data['t']
    y = data['y']
    yerr = data['yerr']
    
    # Create a grid of time points for prediction
    t_min = t.min()
    t_max = t.max()
    t_grid = np.linspace(t_min, t_max, n_grid).reshape(-1, 1)
    
    # Predict mean and standard deviation from the GP
    y_pred, y_std = gp.predict(t_grid, return_std=True)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot original data points with error bars
    plt.errorbar(t.flatten(), y.flatten(), yerr=yerr.flatten(), fmt='o', label='Train', alpha=0.5)

    if heldout_data is not None:
        plt.errorbar(
            np.asarray(heldout_data['t']).flatten(),
            np.asarray(heldout_data['y']).flatten(),
            yerr=np.asarray(heldout_data['yerr']).flatten(),
            fmt='s',
            label='Held-out',
            color='black',
            alpha=0.8,
        )
    
    # Plot GP mean prediction
    plt.plot(t_grid.flatten(), y_pred.flatten(), label='GP Mean', color='red')
    
    # Fill between mean ± std
    plt.fill_between(t_grid.flatten(), (y_pred - y_std).flatten(), (y_pred + y_std).flatten(), 
                     color='red', alpha=0.3, label='GP Std Dev')
    
    plt.title(
        f"{data['obj_type']} | band {data['band']} | object {data['obj_id']}"
    )
    plt.xlabel('Time (days)')
    plt.ylabel('Normalized Flux')
    plt.legend()
    plt.grid()
    plt.show()
