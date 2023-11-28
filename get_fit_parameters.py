import numpy as np
from scipy.optimize import curve_fit

def fitfun(x: np.ndarray, b: float, alpha_cdf: float) -> np.ndarray:
    """
    The function to fit the data to, representing the complementary cumulative distribution function (CCDF).

    Parameters:
    - x (numpy array): The input values.
    - b (float): Scale parameter.
    - alpha_cdf (float): Shape parameter.

    Returns:
    - numpy array: The calculated values based on the input parameters.
    """
    return np.log(b) - alpha_cdf * x

def get_fit_parameters(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Fit the data to the specified function and return the optimized parameters.

    Parameters:
    - x (numpy array): The input values.
    - y (numpy array): The corresponding output values.

    Returns:
    - tuple: Optimized parameters (b, alpha).
    """
    # Using curve_fit from scipy to fit the data to the specified function
    popt, _ = curve_fit(fitfun, x, np.log(y), bounds=([0, 0], [np.inf, np.inf]), p0=[1, 1])

    # Extracting optimized parameters
    b = popt[0]
    alpha = popt[1]
    
    return b, alpha

def fit_parameters(degrees: np.ndarray) -> tuple:
    """
    Fit CCDF data to a power-law distribution and return the optimized parameters.

    Parameters:
    - degrees (numpy array): Input data representing degrees.

    Returns:
    - tuple: Optimized parameters (b, alpha), normalized probabilities (N), and bin centers (centers).
    """
    # Calculate the histogram
    N, edges = np.histogram(np.log(degrees), bins=35)

    # Convert to CCDF
    N = np.array([np.sum(N[i:]) for i in range(len(N))])

    # Remove y=0 probabilities
    nonzero_indices = np.where(N != 0)
    N = N[nonzero_indices]
    edges = edges[:-1][nonzero_indices]

    # Calculate bin centers
    centers = (edges[:-1] + edges[1:]) / 2
    
    # Normalize probabilities
    N = N[:-1]
    N = N / np.sum(N)
    
    # Get optimized parameters using the fit function
    b, alpha = get_fit_parameters(centers, N)
    
    return b, alpha, N, centers