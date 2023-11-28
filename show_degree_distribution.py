import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import networkx as nx
import random

from get_fit_parameters import fitfun, fit_parameters
    
def get_plot(b: float, alpha: float, N: np.ndarray, centers: np.ndarray, ax=None, fig=None, chi=None, args=None) -> tuple:
    """
    Plot the CCDF and the fitted curve for a power-law distribution.

    Parameters:
    - b (float): Scale parameter.
    - alpha (float): Shape parameter.
    - N (numpy array): Normalized probabilities.
    - centers (numpy array): Bin centers.
    - ax (matplotlib.axes._axes.Axes): Optional, the Axes on which to plot.
    - fig (matplotlib.figure.Figure): Optional, the Figure on which to plot.
    - chi (float): Ratio of death to birth rates.

    Returns:
    - tuple: The Figure and Axes objects.
    """
    # Generate a random color for the plot
    color = (random.random(), random.random(), random.random())

    # Plot the CCDF
    ax.plot(np.exp(centers), N, '-*', markersize=10, linewidth=3, color=color)

    # Plot the fitted curve
    size, time_period = args
    fit_curve = np.exp(fitfun(centers, b, alpha))
    ax.plot(np.exp(centers), fit_curve, '-', linewidth=3, color=color, alpha=0.7, label= r'$\chi=$' f'${chi:.2f}$, time_period={time_period:.0f}, size={size}')
    
    # Set log scale for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set plot title and axis labels
    ax.set_title(r"Black Swan networks depending on death/birth ratio $\chi$", fontsize=16)
    ax.set_xlabel('k', fontsize=16)
    ax.set_ylabel("CDF", fontsize=16)

    # Set tick label sizes
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    ax.legend()

    return fig, ax

if __name__ == "__main__":

    load_name = os.path.join(sys.path[0], 'data/black_swan.pkl')
    
    df = pd.read_pickle(load_name)
    
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))

    data = []
    for index, row in df.iterrows():
        G = nx.Graph()
        G.add_nodes_from(row['vertices'])
        G.add_edges_from(row['edges'])
        
        if len(G.edges) == 0:
            continue
        
        degrees = [ G.degree(node) for node in G ]
        degrees = sorted(degrees)
        
        b, alpha, N, centers = fit_parameters(degrees)
        
        row['alpha'], row['b'] = alpha, b
        
        chi = row['lambda_death'] / row['lambda_birth']
        args = [row['size'], row['time_period']]

        fig, ax = get_plot(b, alpha, N, centers, ax=ax1, fig=fig, chi=chi, args=args)
        
        data.append(row)

    df = pd.DataFrame(data)        

    save_name = os.path.join(sys.path[0], 'data/black_swan.csv')
    df.to_csv(save_name, index=False)

    save_name = os.path.join(sys.path[0], 'data/black_swan.pkl')    
    df.to_pickle(save_name)
    
    save_name = os.path.join(sys.path[0], 'figures/power_law_fit.pdf')
    plt.savefig(save_name, dpi=600, transparent=True, bbox_inches='tight')

    plt.show()
