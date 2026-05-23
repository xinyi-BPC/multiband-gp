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
                     color='red', alpha=0.3, label='GP Latent Std Dev')
    
    plt.title(
        f"{data['obj_type']} | band {data['band']} | object {data['obj_id']}"
    )
    plt.xlabel('Time (days)')
    plt.ylabel('Normalized Flux')
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_largest_standardized_residual_cases(object_results, cases, max_plots=10, n_grid=300):
    """
    Plot objects for the largest standardized residual cases.

    object_results must contain "gp", "train_data", and "heldout_data" entries
    for each object, as added in the notebook after evaluate_heldout_metrics.
    """
    plotted = set()
    n_plotted = 0

    for case in cases:
        result_idx = case["result_idx"]
        point_idx = case["point_idx"]
        if (result_idx, point_idx) in plotted:
            continue

        result = object_results[result_idx]
        gp = result["gp"]
        train_data = result["train_data"]
        heldout_data = result["heldout_data"]

        t_all = np.concatenate([
            np.asarray(train_data["t"]),
            np.asarray(heldout_data["t"]),
        ])
        t_grid = np.linspace(t_all.min(), t_all.max(), n_grid).reshape(-1, 1)
        y_pred, y_std = gp.predict(t_grid, return_std=True)

        plt.figure(figsize=(10, 6))
        plt.errorbar(
            np.asarray(train_data["t"]).flatten(),
            np.asarray(train_data["y"]).flatten(),
            yerr=np.asarray(train_data["yerr"]).flatten(),
            fmt="o",
            label="Train",
            alpha=0.5,
        )
        plt.errorbar(
            np.asarray(heldout_data["t"]).flatten(),
            np.asarray(heldout_data["y"]).flatten(),
            yerr=np.asarray(heldout_data["yerr"]).flatten(),
            fmt="s",
            label="Held-out",
            color="black",
            alpha=0.65,
        )
        plt.scatter(
            [case["time"]],
            [case["y"]],
            s=140,
            facecolors="none",
            edgecolors="orange",
            linewidths=2.5,
            label="Highlighted |z| case",
            zorder=5,
        )
        plt.plot(t_grid.flatten(), y_pred.flatten(), label="GP Mean", color="red")
        plt.fill_between(
            t_grid.flatten(),
            (y_pred - y_std).flatten(),
            (y_pred + y_std).flatten(),
            color="red",
            alpha=0.25,
            label="GP Latent Std Dev",
        )
        plt.axvline(case["train_time_min"], color="gray", linestyle="--", alpha=0.4)
        plt.axvline(case["train_time_max"], color="gray", linestyle="--", alpha=0.4)
        plt.title(
            f"{case['object_id']} | band {case['band']} | "
            f"z={case['z']:.2f} | edge={case['outside_train_range']} | peak={case['near_peak']}"
        )
        plt.xlabel("Time (standardized, peak-aligned)")
        plt.ylabel("Normalized Flux")
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        plt.grid()
        plt.tight_layout()
        plt.show()

        plotted.add((result_idx, point_idx))
        n_plotted += 1
        if n_plotted >= max_plots:
            break
