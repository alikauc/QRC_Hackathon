"""
Quantum Reservoir Computing - Python Implementation
Converted from Julia implementation in Time_series.jl

This module provides a classical approximation of quantum reservoir computing
using Echo State Network principles.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from typing import List, Tuple

# ============================================================================
# Data Normalization Constants
# ============================================================================
Max_RV = -1.2543188032019446
Min_RV = -4.7722718186046515
coe = (Max_RV - Min_RV)**2
dif = Max_RV - Min_RV


# ============================================================================
# Model Structure
# ============================================================================
class MyModel:
    """
    Custom model structure for reservoir computing.
    
    Attributes:
        L: Number of reservoir features/neurons
        OutLen: Length of output prediction
        W: Weight matrix for readout layer (L × OutLen)
        Features: Names of input features used
    """
    def __init__(self, L: int, OutLen: int, Features: List[str]):
        self.L = L
        self.OutLen = OutLen
        self.W = np.zeros((L, OutLen))
        self.Features = Features


# ============================================================================
# Data Preprocessing Functions
# ============================================================================
def normalization(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Normalize data to range [a, b]."""
    xmax = np.max(x)
    xmin = np.min(x)
    return (b - a) * (x - xmin) / (xmax - xmin) + a


def denormalization(x1: float, x2: float, y: np.ndarray, a: float, b: float) -> np.ndarray:
    """Denormalize data from range [a,b] back to original range [x2,x1]."""
    xmax = x1
    xmin = x2
    return (y - a) * (xmax - xmin) / (b - a) + xmin


def denormalization_alt(x: np.ndarray, y: np.ndarray, a: float, b: float) -> np.ndarray:
    """Denormalize data from range [a, b] back to original scale of x."""
    xmax = np.max(x)
    xmin = np.min(x)
    return (y - a) * (xmax - xmin) / (b - a) + xmin


# ============================================================================
# Evaluation Metrics
# ============================================================================
def MAPE(x: np.ndarray, y: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    return np.mean(np.abs((x - y) * dif / ((y + 1) * dif + Min_RV))) * 100


def MAPE_std(x: np.ndarray, y: np.ndarray) -> float:
    """Standard deviation of Absolute Percentage Error."""
    return np.std(np.abs((x - y) * dif / ((y + 1) * dif + Min_RV))) * 100


def MAE(x: np.ndarray, y: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs((x - y) * dif))


def MAE_std(x: np.ndarray, y: np.ndarray) -> float:
    """Standard deviation of Absolute Error."""
    return np.std(np.abs((x - y) * dif))


def MSE(x: np.ndarray, y: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean(((x - y)**2) * coe)


def MSE_std(x: np.ndarray, y: np.ndarray) -> float:
    """Standard deviation of Squared Error."""
    return np.std(((x - y)**2) * coe)


def RMSE(x: np.ndarray, y: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean(((x - y)**2) * coe))


def wave(y: np.ndarray) -> np.ndarray:
    """
    Compute the directional change (wave) of a time series.
    Returns +1 if increasing, -1 if decreasing.
    """
    L = len(y)
    w = np.zeros(L - 1)
    for i in range(L - 1):
        w[i] = np.sign(y[i + 1] - y[i])
    return w


def hitrate(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the hit rate (directional accuracy) between predictions and actuals."""
    L = len(x)
    x = np.concatenate([[-0.5704088242386152], x])
    y = np.concatenate([[-0.5704088242386152], y])
    wx = wave(x)
    wy = wave(y)
    return np.sum(wx == wy) / L


# ============================================================================
# Loss Functions for Volatility Forecasting
# ============================================================================
def compute_qlike(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """
    Compute the QLIKE (Quasi-Likelihood) loss function for volatility forecasting.
    """
    forecasts = np.abs((forecasts + 1) * dif + Min_RV)
    actuals = np.abs((actuals + 1) * dif + Min_RV)
    ratio = actuals / forecasts
    qlike = np.sum(ratio - np.log(ratio) - 1)
    return qlike


def compute_qlike2(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """Alternative QLIKE computation using exponential transformation."""
    forecasts = (forecasts + 1) * dif + Min_RV
    actuals = (actuals + 1) * dif + Min_RV
    ratio = np.exp(actuals) / np.exp(forecasts)
    qlike = np.sum(ratio - (actuals - forecasts) - 1)
    return qlike


# ============================================================================
# Utility Functions
# ============================================================================
def coeff_matrix(N: int, J: float) -> np.ndarray:
    """Generate a random symmetric coupling matrix for the reservoir."""
    m = np.random.rand(N, N)
    m = (m + m.T) / 2
    np.fill_diagonal(m, 0.0)
    eigvals = np.linalg.eigvals(m)
    return m / np.max(eigvals) * J


def shift(V: np.ndarray, step: int) -> np.ndarray:
    """Shift a vector or matrix forward by 'step' positions, padding with zeros."""
    if V.ndim == 1:
        V1 = np.zeros(len(V))
        V1[step:] = V[:-step]
    else:
        V1 = np.zeros_like(V)
        V1[step:, :] = V[:-step, :]
    return V1


def rolling(V: np.ndarray, window: int) -> np.ndarray:
    """Create a rolling window matrix from a vector."""
    M = np.zeros((len(V), window))
    for i in range(window):
        M[i:, i] = V[:len(V) - i]
    return M


# ============================================================================
# Reservoir Computing Functions
# ============================================================================
def create_reservoir(nqubit: int, coupling_matrix: np.ndarray, 
                    input_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a classical reservoir as an approximation of the quantum reservoir.
    Uses Echo State Network principles.
    """
    reservoir_size = 2**nqubit
    
    # Initialize reservoir weights
    W_reservoir = np.random.randn(reservoir_size, reservoir_size) * 0.1
    
    # Normalize spectral radius for stability
    spectral_radius = 0.95
    eigvals = np.linalg.eigvals(W_reservoir)
    W_reservoir = W_reservoir * (spectral_radius / np.max(np.abs(eigvals)))
    
    # Input weights
    W_in = np.random.randn(reservoir_size, input_size) * 0.5
    
    return W_reservoir, W_in


def reservoir_state_update(state: np.ndarray, W_reservoir: np.ndarray, 
                          W_in: np.ndarray, input_data: np.ndarray, 
                          tau: float = 1.0) -> np.ndarray:
    """Update reservoir state with new input."""
    new_state = np.tanh(W_reservoir @ state + W_in @ input_data)
    return new_state


def Quantum_Reservoir(Data: pd.DataFrame, features: List[str], 
                     coupling_matrix: np.ndarray, nqubit: int, 
                     K_delay: int, VirtualNode: int, tau: float) -> np.ndarray:
    """
    Classical approximation of quantum reservoir computing.
    
    This function simulates the quantum reservoir using classical reservoir computing.
    
    Args:
        Data: Input time series data
        features: List of feature column names
        coupling_matrix: Reservoir coupling matrix
        nqubit: Number of qubits (reservoir dimension = 2^nqubit)
        K_delay: Number of time delay steps
        VirtualNode: Number of virtual nodes
        tau: Evolution time parameter
        
    Returns:
        Output matrix (nqubit*VirtualNode × L) of reservoir states
    """
    L = len(Data)
    InputSize = len(features)
    reservoir_size = 2**nqubit
    
    # Create reservoir
    W_reservoir, W_in = create_reservoir(nqubit, coupling_matrix, InputSize)
    
    # Output features: nqubit * VirtualNode
    Output = np.zeros((nqubit * VirtualNode, L))
    
    # Process each time step
    for l in range(K_delay, L):
        # Initialize reservoir state
        state = np.zeros(reservoir_size)
        
        # Process K_delay previous time steps
        for k in range(K_delay, 0, -1):
            # Extract input features
            input_data = np.array([Data[features[i]].iloc[l - k] for i in range(InputSize)])
            
            # Normalize input to [-1, 1] range
            input_data = np.clip(input_data, -1, 1)
            
            # Update reservoir state
            state = reservoir_state_update(state, W_reservoir, W_in, input_data, tau)
        
        # Virtual nodes: sample reservoir state multiple times with small evolution
        for v in range(VirtualNode):
            # Small evolution step
            state = reservoir_state_update(
                state, W_reservoir, 
                np.zeros((reservoir_size, InputSize)), 
                np.zeros(InputSize), 
                tau / VirtualNode
            )
            
            # Extract features (measurements)
            for n in range(nqubit):
                idx = v * nqubit + n
                start_idx = n * (reservoir_size // nqubit)
                end_idx = (n + 1) * (reservoir_size // nqubit)
                Output[idx, l] = np.mean(state[start_idx:end_idx])
    
    return Output


# ============================================================================
# Training Function
# ============================================================================
def train_model(signal: np.ndarray, Data: pd.DataFrame, 
               model: MyModel, L: int, OutLen: int, 
               Total: int, ws: int = 0) -> Tuple[np.ndarray, MyModel]:
    """
    Train the reservoir model using ridge regression.
    
    Args:
        signal: Reservoir output signals
        Data: Original data with target variable
        model: Model structure
        L: Number of out-of-sample predictions
        OutLen: Output dimension
        Total: Total number of data points
        ws: Window start index
        
    Returns:
        Predictions and trained model
    """
    wi = Total - L
    W_paras = np.zeros((L, OutLen))
    Pre = np.zeros(L)
    
    for j in range(L):
        # Training data
        y_train = Data['RV'].iloc[ws + j:ws + wi + j].values
        x_train = signal[:, ws + j:ws + wi + j]
        
        # Ridge regression
        ridge = Ridge(alpha=1e-8)
        ridge.fit(x_train.T, y_train)
        W_paras[j, :] = ridge.coef_[:OutLen]
        
        # Prediction
        Pre[j] = np.dot(W_paras[j, :], signal[:, ws + wi + j])
        model.W[j, :] = W_paras[j, :]
    
    return Pre, model


def evaluate_model(predictions: np.ndarray, actuals: np.ndarray) -> dict:
    """
    Evaluate model performance with multiple metrics.
    
    Returns:
        Dictionary with all evaluation metrics
    """
    return {
        'hitrate': hitrate(predictions, actuals),
        'MSE': MSE(predictions, actuals),
        'RMSE': RMSE(predictions, actuals),
        'MAE': MAE(predictions, actuals),
        'MAPE': MAPE(predictions, actuals),
        'QLIKE': compute_qlike(predictions, actuals)
    }


if __name__ == "__main__":
    print("QRC Python Module - Classical Approximation")
    print("=" * 60)
    print("This module provides quantum reservoir computing functions")
    print("converted from Julia implementation.")
    print("\nUsage:")
    print("  from qrc_python import Quantum_Reservoir, train_model")
    print("  # ... see Time_serial_Finance_regression.ipynb for examples")
