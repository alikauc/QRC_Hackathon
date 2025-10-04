# Quantum Reservoir Computing for Realized Volatility Forecasting

This repository contains the implementation of a Quantum Reservoir Computing (QRC) approach for forecasting realized volatility in financial time series. The project leverages quantum computing techniques to improve volatility predictions compared to classical reservoir computing and LSTM models.

## Overview

The project implements and compares three approaches for realized volatility forecasting:
1. **Quantum Reservoir Computing (QRC)** - Main implementation using quantum circuits
2. **Classical Reservoir Computing** - Traditional Echo State Networks (ESN)
3. **Long Short-Term Memory (LSTM)** - Deep learning baseline

## Repository Structure

- **`Time_series.jl`** - Core Julia implementation of the Quantum Reservoir Computing model
- **`time_series.py`** - Python conversion of the Julia implementation (new!)
- **`Time_serial_Finance_regression.ipynb`** - Main notebook demonstrating the QRC approach for financial time series
- **`Reservoir_Learning.ipynb`** - Quantum reservoir learning experiments and visualizations
- **`classical_reservoir.ipynb`** - Classical reservoir computing baseline implementation
- **`LSTM.ipynb`** - LSTM neural network baseline implementation
- **`Data.CSV`** - Financial time series data with macroeconomic features
- **`predict_result.csv`** - Model predictions output
- **`coeff_10.jld2`** - Pre-computed coefficients for the quantum reservoir
- **`requirements.txt`** - Python package dependencies

## Data Features

The dataset includes the following features for realized volatility forecasting:
- **RV** - Realized Volatility (target variable)
- **MKT** - Market return
- **DP** - Dividend price ratio
- **EP** - Earnings price ratio
- **IP** - Industrial production
- **DEF** - Default spread
- **SMB** - Size factor (Small Minus Big)
- **HML** - Value factor (High Minus Low)
- **TB** - Treasury bill rate
- **INF** - Inflation rate
- **STR** - Short-term reversal

## Requirements

### Julia Packages
- `VQC` - Variational Quantum Computing framework
- `QuantumCircuits` - Quantum circuit manipulation
- `Flux` - Machine learning framework
- `CUDA` - GPU acceleration
- `Statistics`, `StatsBase`, `LinearAlgebra` - Mathematical operations

### Python Packages
- `numpy` - Numerical computing
- `scipy` - Scientific computing (eigenvalues, linear algebra)
- `pandas` - Data manipulation
- `torch` - PyTorch for LSTM
- `reservoirpy` - Classical reservoir computing
- `matplotlib` - Visualization

#### Optional Python Packages
- `cupy` - GPU acceleration (CUDA)
- `qiskit` / `cirq` / `pennylane` / `qutip` - Quantum computing libraries (for actual quantum operations)

Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Key Methods

### Quantum Reservoir Computing
The QRC implementation includes:
- **Virtual Nodes** - Time-multiplexing for increased reservoir dimensionality
- **Quantum State Evolution** - Using parameterized quantum circuits
- **Density Matrix Operations** - Both CPU and GPU-accelerated implementations
- **Measurement** - Observable expectations as reservoir outputs

### Evaluation Metrics
- **MAPE** - Mean Absolute Percentage Error
- **MAE** - Mean Absolute Error
- **MSE** - Mean Squared Error
- **RMSE** - Root Mean Squared Error
- **Hit Rate** - Directional accuracy of predictions
- **QLIKE** - Quasi-Likelihood loss for volatility forecasting

## Usage

### Python Implementation

The Python version (`time_series.py`) provides the same functionality as the Julia implementation with standard Python libraries:

```python
# Import the module
from time_series import *

# Test utility functions
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
normalized = normalization(data, -1, 1)
print(f"Normalized data: {normalized}")

# Generate coupling matrix for quantum reservoir
nqubit = 4
J = 1.0
ps = coeff_matrix(nqubit, J)

# Evaluate predictions
predictions = np.array([0.1, 0.2, 0.15, 0.25, 0.3])
actuals = np.array([0.12, 0.18, 0.16, 0.24, 0.28])
print(f"MSE: {MSE(predictions, actuals):.6f}")
print(f"RMSE: {RMSE(predictions, actuals):.6f}")
print(f"Hit rate: {hitrate(predictions, actuals):.4f}")
```

**Important Note:** The quantum computing operations in the Python version are placeholder implementations. For actual quantum reservoir computing with Python, you need to integrate with a quantum computing library such as:
- **Qiskit** (IBM Quantum): https://qiskit.org/
- **Cirq** (Google): https://quantumai.google/cirq
- **PennyLane** (Xanadu): https://pennylane.ai/
- **QuTiP** (Quantum Toolbox): https://qutip.org/

The utility functions, evaluation metrics, and data preprocessing functions are fully functional.

### Julia Implementation (Original)

### Running the Quantum Reservoir Model

```julia
# Load the Time_series.jl file
include("Time_series.jl")

# Set up quantum reservoir parameters
nqubit = 4  # Number of qubits
VirtualNode = 10  # Number of virtual nodes
K_delay = 3  # Time delay steps
τ = 1.0  # Evolution time

# Create quantum Hamiltonian
ps = coeff_matrix(nqubit, J)  # Generate coupling matrix
QR = Qreservoir(nqubit, ps)

# Define observables
Observable = [QubitsTerm(i=>"X") for i in 1:nqubit]

# Process data with quantum reservoir
features = ["RV", "MKT", "DP", "EP", "IP", "DEF", "SMB", "TB", "HML", "INF", "STR"]
Output = Quantum_Reservoir(Data, features, QR, Observable, K_delay, VirtualNode, τ, nqubit)
```

### Running Classical Baselines

For classical reservoir computing, see `classical_reservoir.ipynb`.
For LSTM baseline, see `LSTM.ipynb`.

## Model Architecture

The Quantum Reservoir Computing model consists of:
1. **Input Encoding** - Financial features encoded into qubit rotation gates
2. **Reservoir Layer** - Quantum state evolution with Hamiltonian dynamics
3. **Readout Layer** - Linear regression on quantum measurements
4. **Rolling Window** - Online learning with sliding window approach

## Results

The model performs multi-step ahead forecasting of realized volatility using a rolling window approach:
- Training window: 571 samples
- Test samples: 245 predictions
- Total dataset: 816 time points (1950-2017)

Performance is evaluated using multiple metrics including MAPE, MAE, RMSE, and directional accuracy (hit rate).

## Citation

If you use this code in your research, please cite:

```
Quantum Reservoir Computing for Realized Volatility Forecasting
[Add publication details when available]
```

## License

[Add license information]

## Acknowledgments

This project was developed as part of a quantum computing hackathon focused on applying quantum machine learning to financial forecasting problems.
