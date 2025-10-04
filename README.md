# Quantum Reservoir Computing for Realized Volatility Forecasting

This repository contains the implementation of a Quantum Reservoir Computing (QRC) approach for forecasting realized volatility in financial time series. The project leverages quantum computing techniques to improve volatility predictions compared to classical reservoir computing and LSTM models.

> **🆕 NEW: Python Version Available!** The main notebook `Time_serial_Finance_regression.ipynb` has been converted from Julia to Python. The conversion includes a classical reservoir computing approximation that maintains the same functionality. See `CONVERSION_NOTES.md` for details.

## Overview

The project implements and compares three approaches for realized volatility forecasting:
1. **Quantum Reservoir Computing (QRC)** - Main implementation using quantum circuits
2. **Classical Reservoir Computing** - Traditional Echo State Networks (ESN)
3. **Long Short-Term Memory (LSTM)** - Deep learning baseline

## Repository Structure

- **`Time_series.jl`** - Core Julia implementation of the Quantum Reservoir Computing model
- **`Time_serial_Finance_regression.ipynb`** - **[NEW: Python Version]** Main notebook demonstrating the QRC approach (converted to Python)
- **`qrc_python.py`** - **[NEW]** Python module with all QRC functions (can be imported and reused)
- **`CONVERSION_NOTES.md`** - **[NEW]** Detailed documentation of Julia to Python conversion
- **`Reservoir_Learning.ipynb`** - Quantum reservoir learning experiments and visualizations
- **`classical_reservoir.ipynb`** - Classical reservoir computing baseline implementation
- **`LSTM.ipynb`** - LSTM neural network baseline implementation
- **`Data.CSV`** - Financial time series data with macroeconomic features
- **`predict_result.csv`** - Model predictions output
- **`coeff_10.jld2`** - Pre-computed coefficients for the quantum reservoir

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

### Python Packages (for notebooks)
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning (Ridge regression)
- `scipy` - Scientific computing
- `matplotlib` - Visualization
- `torch` - PyTorch for LSTM (LSTM.ipynb only)
- `reservoirpy` - Classical reservoir computing
- `matplotlib` - Visualization

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

### Running the Quantum Reservoir Model (Python - NEW!)

```python
# Import the QRC module
import qrc_python as qrc
import pandas as pd
import numpy as np

# Load data
Data = pd.read_csv("Data.CSV")

# Set up quantum reservoir parameters
nqubit = 10  # Number of qubits
VirtualNode = 1  # Number of virtual nodes
K_delay = 3  # Time delay steps
tau = 1.0  # Evolution time

# Generate coupling matrix
coupling_matrix = qrc.coeff_matrix(nqubit, 1.0)

# Process data with quantum reservoir (classical approximation)
features = ["RV", "MKT", "DP", "IP", "RV_q", "STR", "DEF"]
signal = qrc.Quantum_Reservoir(Data, features, coupling_matrix, 
                               nqubit, K_delay, VirtualNode, tau)

# Train model and make predictions
# ... (see Time_serial_Finance_regression.ipynb for full example)
```

**Or simply run the Jupyter notebook:**
```bash
jupyter notebook Time_serial_Finance_regression.ipynb
```

### Running the Quantum Reservoir Model (Julia - Original)

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
