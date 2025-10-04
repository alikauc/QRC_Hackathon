# ============================================================================
# Quantum Reservoir Computing for Time Series Forecasting
# ============================================================================
# This file implements a quantum reservoir computing (QRC) approach for 
# financial time series forecasting, specifically for realized volatility prediction.
#
# Key Components:
# - Quantum circuit operations using VQC and QuantumCircuits packages
# - CUDA-accelerated density matrix operations
# - Reservoir computing with virtual nodes
# - Multiple evaluation metrics for forecasting performance
# ============================================================================

# Add custom package paths for quantum computing libraries
push!(LOAD_PATH,"../package/QuantumCircuits_demo/src","../package/VQC_demo_cuda/src")

# Import required packages
using VQC, VQC.Utilities                    # Variational Quantum Computing framework
using QuantumCircuits, QuantumCircuits.Gates # Quantum circuit manipulation
using Flux:train!                            # Machine learning training utilities
using Flux                                   # ML framework
using Random                                 # Random number generation
using Statistics                             # Statistical functions
using StatsBase                              # Extended statistics
using LinearAlgebra                          # Linear algebra operations
using CUDA                                   # GPU acceleration
import LinearAlgebra: tr                     # Import trace function for extension

# ============================================================================
# Model Structure Definition
# ============================================================================

"""
    MyModel

Custom model structure for reservoir computing.

# Fields
- `L::Int` - Number of reservoir features/neurons
- `OutLen::Int` - Length of output prediction
- `W::Matrix{Float64}` - Weight matrix for readout layer (L × OutLen)
- `Features::Vector{String}` - Names of input features used

The model stores the learned weights that map reservoir states to predictions.
"""
struct MyModel
    L::Int
    OutLen::Int
    W::Matrix{Float64}
    Features::Vector{String}
end

# Constructor: Initialize model with zero weights
MyModel(L::Int,OutLen::Int,Features::Vector{String},) = MyModel(L,OutLen,zeros(L,OutLen),Features)

# ============================================================================
# CUDA Density Matrix Operations
# ============================================================================

"""
    tr(m::CuDensityMatrixBatch)

Compute the trace of a CUDA density matrix batch.

This function extends LinearAlgebra.tr to handle batched density matrices on GPU.
The trace is computed for quantum states represented as density matrices.

# Arguments
- `m::CuDensityMatrixBatch` - Batched density matrix on GPU

# Returns
- Trace of the first item in the batch (scalar value)

# Note
Currently returns only the trace of the first density matrix in the batch.
"""
function tr(m::CuDensityMatrixBatch)
    mat = storage(m)
    # Reshape to separate batch dimension: (2^n × 2^n × batch_size)
    remat = reshape(mat,2^m.nqubits,2^m.nqubits,m.nitems)
    x=zeros(eltype(m),m.nitems)
    # Compute trace for each item in batch
    for i in 1:m.nitems
        x[i]=CUDA.tr(remat[:,:,i])
    end
    return x[1]
end


# ============================================================================
# Data Normalization Constants
# ============================================================================
# These constants are derived from the realized volatility (RV) data range
# and are used for denormalization and metric calculations.

# Maximum and minimum values of realized volatility in the raw dataset
Max_RV=-1.2543188032019446
Min_RV=-4.7722718186046515

# Coefficient for MSE calculation: (max - min)^2
coe = (Max_RV-Min_RV)^2

# Difference for scaling: (max - min)
dif = (Max_RV-Min_RV)

# ============================================================================
# Data Preprocessing Functions
# ============================================================================

"""
    denormalization(x1, x2, y, a, b)

Denormalize data from range [a,b] back to original range [x2,x1].

# Arguments
- `x1` - Original maximum value (xmax)
- `x2` - Original minimum value (xmin)
- `y` - Normalized data in range [a,b]
- `a` - Lower bound of normalized range
- `b` - Upper bound of normalized range

# Returns
- Denormalized data in original scale [x2,x1]

This is used to convert normalized predictions back to the original data scale.
"""
function denormalization(x1,x2,y,a,b)
    xmax=x1
    xmin=x2
    f(z)=(z-a)*(xmax-xmin)/(b-a)+xmin
    return map(f,y)
end

# ============================================================================
# Tensor Product Operations
# ============================================================================

"""
    ⊗(A::DensityMatrix, B::DensityMatrix)

Compute the tensor product (Kronecker product) of two density matrices.

This operation creates a composite quantum state from two subsystems.
Symbol: ⊗ can be typed as \\otimes<TAB> in Julia.

# Arguments
- `A::DensityMatrix` - First density matrix
- `B::DensityMatrix` - Second density matrix

# Returns
- Combined density matrix representing the tensor product state
"""
function ⊗(A::DensityMatrix,B::DensityMatrix)
    return DensityMatrix(kron(storage(A),storage(B)),nqubits(A)+nqubits(B))
end

"""
    ⊗(A::CuDensityMatrixBatch, B::CuDensityMatrixBatch)

GPU-accelerated tensor product of batched density matrices.

# Arguments
- `A::CuDensityMatrixBatch` - First batched density matrix on GPU
- `B::CuDensityMatrixBatch` - Second batched density matrix on GPU

# Returns
- Combined batched density matrix on GPU
"""
function ⊗(A::CuDensityMatrixBatch,B::CuDensityMatrixBatch)
    return CuDensityMatrixBatch(kron(storage(A),storage(B)),A.nqubits+B.nqubits,1)
end

# ============================================================================
# Matrix Multiplication Overloads
# ============================================================================
# These functions extend the * operator to work with density matrices
# and enable quantum state evolution: ρ' = U * ρ * U†

import Base: *

"""
    *(A::Union{Matrix,CuArray}, B::DensityMatrix)

Left multiplication of density matrix by a unitary operator.
Represents quantum evolution: ρ → U*ρ
"""
function *(A::Union{Matrix,CuArray}, B::DensityMatrix)
    return DensityMatrix(Array(A*CuArray(storage(B))))
end

"""
    *(B::DensityMatrix, A::Union{CuArray,Adjoint})

Right multiplication of density matrix by a unitary operator.
Represents quantum evolution: ρ → ρ*U†
"""
function *(B::DensityMatrix,A::Union{CuArray,Adjoint})
    return DensityMatrix(Array(CuArray(storage(B))*A))
end

"""
    *(A::Union{Matrix,CuArray}, B::CuDensityMatrixBatch)

GPU-accelerated left multiplication for batched density matrices.
"""
function *(A::Union{Matrix,CuArray}, B::CuDensityMatrixBatch)
    return CuDensityMatrixBatch(A*storage(B),B.nqubits,B.nitems)
end

"""
    *(B::CuDensityMatrixBatch, A::Union{CuArray,Adjoint})

GPU-accelerated right multiplication for batched density matrices.
"""
function *(B::CuDensityMatrixBatch,A::Union{CuArray,Adjoint})
    return CuDensityMatrixBatch(storage(B)*A,B.nqubits,B.nitems)
end

# ============================================================================
# Quantum Reservoir Hamiltonian
# ============================================================================

"""
    Qreservoir(nqubit, ps)

Construct the Hamiltonian for the quantum reservoir.

The Hamiltonian consists of:
1. XX coupling terms between all qubit pairs (interaction)
2. Z terms on each qubit (local field)

This creates a recurrent quantum system suitable for reservoir computing.

# Arguments
- `nqubit::Int` - Number of qubits in the reservoir
- `ps::Matrix` - Coupling strength matrix (symmetric, nqubit × nqubit)

# Returns
- `H::QubitsOperator` - Hamiltonian operator defining reservoir dynamics

# Mathematical Form
H = Σᵢⱼ ps[i,j] XᵢXⱼ + Σᵢ Zᵢ

where Xᵢ and Zᵢ are Pauli X and Z operators on qubit i.
"""
function Qreservoir(nqubit,ps)
    H=QubitsOperator()
    # Add XX coupling terms between all pairs of qubits
    for i in 1:nqubit
        for j in i+1:nqubit
            H+=QubitsTerm(i=>"X",j=>"X",coeff=ps[i,j])
        end
    end
    
    # Add local Z field on each qubit
    for i in 1:nqubit
        H+=QubitsTerm(i=>"Z",coeff=1)
    end
    return H
end

"""
    normalization(x, a, b)

Normalize data to range [a, b].

# Arguments
- `x` - Input data vector
- `a` - Target minimum value
- `b` - Target maximum value

# Returns
- Normalized data scaled to [a, b]
"""
function normalization(x,a,b)
    xmax=maximum(x)
    xmin=minimum(x)
    f(z)=(b-a)*(z-xmin)/(xmax-xmin)+a
    return map(f,x)
end

"""
    denormalization(x, y, a, b)

Denormalize data from range [a, b] back to original scale of x.

# Arguments
- `x` - Original data (used to determine scale)
- `y` - Normalized data to be denormalized
- `a` - Lower bound of normalized range
- `b` - Upper bound of normalized range

# Returns
- Denormalized data in original scale of x
"""
function denormalization(x,y,a,b)
    xmax=maximum(x)
    xmin=minimum(x)
    f(z)=(z-a)*(xmax-xmin)/(b-a)+xmin
    return map(f,y)
end


# ============================================================================
# Evaluation Metrics for Forecasting
# ============================================================================
# These metrics evaluate the quality of realized volatility predictions.
# All metrics account for the denormalization of the data using global constants.

"""Mean Absolute Percentage Error - measures relative prediction error"""
MAPE(x,y) = mean(abs.((x-y).*dif./((y.+1)*dif.+Min_RV)))*100

"""Standard deviation of Absolute Percentage Error"""
MAPE_std(x,y) = std(abs.((x-y).*dif./((y.+1)*dif.+Min_RV)))*100

"""Mean Absolute Error - measures average absolute prediction error"""
MAE(x,y) = mean(abs.((x-y).*dif))

"""Standard deviation of Absolute Error"""
MAE_std(x,y) = std(abs.((x-y).*dif))

"""Mean Squared Error - penalizes large errors more heavily"""
MSE(x,y) = mean(((x-y).^2).*coe)

"""Standard deviation of Squared Error"""
MSE_std(x,y) = std(((x-y).^2).*coe)

"""Root Mean Squared Error - MSE in original scale"""
RMSE(x,y) = sqrt(mean(((x-y).^2).*coe))



# ============================================================================
# Main Quantum Reservoir Computing Function
# ============================================================================

"""
    Quantum_Reservoir(Data, features, QR, Observable, K_delay, VirtualNode, τ, nqubit)

Process time series data through a quantum reservoir computing system.

This is the core function that implements the quantum reservoir computing approach
with virtual nodes for time-multiplexing. It uses GPU acceleration via CUDA.

# Arguments
- `Data` - Input time series data matrix (L × num_features)
- `features` - Vector of feature names to use from Data
- `QR` - Quantum reservoir Hamiltonian operator
- `Observable` - Vector of observables to measure (Pauli operators)
- `K_delay::Int` - Number of time delay steps (memory length)
- `VirtualNode::Int` - Number of virtual nodes for time-multiplexing
- `τ` - Total evolution time for each delay step
- `nqubit::Int` - Total number of qubits in the system

# Returns
- `Output::Matrix` - Reservoir states (N*VirtualNode × L) where N = length(Observable)

# Process
1. For each time step l (from K_delay+1 to L):
   2. Create input encoding circuit with Ry gates
   3. Initialize reservoir state ρᵣ
   4. Process K_delay previous time steps in reverse order:
      a. Encode input features into quantum state
      b. Combine with reservoir state via tensor product
      c. Evolve under Hamiltonian U = exp(-iτH)
      d. Trace out input qubits to update reservoir
   5. For the final step, use virtual nodes:
      a. Evolve with smaller time steps δτ = τ/VirtualNode
      b. Measure observables at each virtual node
   6. Store measurements as reservoir output features

# Virtual Nodes
Virtual nodes increase the effective dimensionality of the reservoir by
time-multiplexing: one physical system creates multiple virtual nodes through
sequential evolution and measurement at intermediate time steps.
"""
function Quantum_Reservoir(Data, features, QR, Observable, K_delay, VirtualNode, τ, nqubit)
    N = length(Observable)          # Number of observables
    L = size(Data,1)                # Total number of time steps
    
    InputSize = length(features)    # Number of input features

    # Initialize output matrix: (N*VirtualNode) features × L time steps
    Output = zeros(N*VirtualNode,L)
    
    # Time step for virtual nodes
    δτ = τ/VirtualNode

    # Precompute evolution operators on GPU
    δU = CuArray(exp(-im*δτ*Matrix(matrix(QR))))  # Small step evolution
    U = CuArray(exp(-im*τ*Matrix(matrix(QR))))    # Full step evolution

    # Process each time step (starting after initial delay period)
    for l in (K_delay+1):L
        # Create parameterized input encoding circuit
        cir = QCircuit()
        for i in 1:InputSize
            push!(cir,RyGate(i,rand(),isparas=true))
        end
        
        # Initialize reservoir state (qubits not used for input)
        ρᵣ = CuDensityMatrixBatch{ComplexF32}(nqubit-InputSize,1)
        
        # Process K_delay previous time steps (reverse chronological order)
        for k in K_delay:-1:1
            # Create input state with current features
            ρ₁ = CuDensityMatrixBatch{ComplexF32}(InputSize,1)
            para = [Data[l-k,str] for str in features]  # Extract feature values
            cir(para.*pi)  # Set circuit parameters (scaled by π)
            
            # Combine reservoir and input states
            if (nqubit-InputSize)==0
                ρ = cir(ρ₁)          # All qubits are input qubits
            else
                ρ = ρᵣ⊗(cir(ρ₁))     # Tensor product of reservoir and input
            end
            
            if k!=1
                # Not the final delay step: evolve and trace out input
                ρ = U*ρ*U'                              # Unitary evolution
                ρᵣ=partial_tr(ρ, Vector(1:InputSize))  # Trace out input qubits
            else
                # Final delay step: use virtual nodes
                it=1
                for v in 1:VirtualNode
                    ρ = δU*ρ*δU'                                    # Small time evolution
                    for n in 1:N 
                        # Measure each observable and store result
                        Output[it,l] = vec(real(expectation(B[n],ρ)))[1]
                        it+=1
                    end
                end
            end
        end
    end
    return Output
end


# ============================================================================
# Loss Functions for Volatility Forecasting
# ============================================================================

"""
    compute_qlike(forecasts, actuals)

Compute the QLIKE (Quasi-Likelihood) loss function for volatility forecasting.

QLIKE is a robust loss function specifically designed for evaluating variance
forecasts. It penalizes both over- and under-prediction asymmetrically.

# Arguments
- `forecasts` - Predicted volatility values (normalized)
- `actuals` - Actual observed volatility values (normalized)

# Returns
- QLIKE loss value (lower is better)

# Formula
QLIKE = Σ(actual/forecast - log(actual/forecast) - 1)

This loss function is widely used in financial econometrics for evaluating
volatility models as it handles the non-negativity and heteroskedasticity
of variance data appropriately.
"""
function compute_qlike(forecasts,  actuals)
    """
    Compute the QLIKE (Quasi-Likelihood) loss function for evaluating forecasting accuracy.
    forecasts: Forecasted variance (sigma squared from a model)
    actuals: Realized variance (actual observed variance)
    """
    # Denormalize: convert from [-1,1] range back to original scale
    forecasts =abs.((forecasts.+1)*dif.+Min_RV)
    actuals = abs.((actuals.+1)*dif.+Min_RV)

    # Calculate the ratio and ensure it's positive
    ratio = actuals ./ forecasts

    # Compute QLIKE: Σ(ratio - log(ratio) - 1)
    qlike = sum(ratio - log.(ratio).-1)
    return qlike
end

"""
    compute_qlike2(forecasts, actuals)

Alternative QLIKE computation using exponential transformation.

This variant uses exponential transformation before computing the ratio,
which can be more stable for certain data distributions.
"""
function compute_qlike2(forecasts, actuals)
    # Denormalize data
    forecasts =(forecasts.+1)*dif.+Min_RV
    actuals = (actuals.+1)*dif.+Min_RV

    # Calculate ratio with exponential transformation
    ratio = exp.(actuals) ./ exp.(forecasts)

    # Compute modified QLIKE
    qlike = sum(ratio -(actuals-forecasts).-1)
    return qlike
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    coeff_matrix(N, J)

Generate a random symmetric coupling matrix for the quantum reservoir.

Creates a random symmetric matrix with zero diagonal, normalized such that
the maximum eigenvalue equals J. This ensures stable quantum dynamics.

# Arguments
- `N::Int` - Matrix dimension (number of qubits)
- `J` - Desired maximum eigenvalue (coupling strength)

# Returns
- Symmetric N×N matrix with max eigenvalue J and zero diagonal

The symmetric structure ensures the Hamiltonian is Hermitian (physical),
and the normalization controls the energy scale of the system.
"""
function coeff_matrix(N,J)
    m=rand(N,N)
    # Symmetrize the matrix
    m=(m+transpose(m))./2
    # Zero out diagonal (no self-interaction)
    for i in 1:N
        m[i,i]=0.0
    end
    # Normalize by largest eigenvalue and scale by J
    return m./max(eigvals(m)...).*J
end

"""
    wave(y)

Compute the directional change (wave) of a time series.

Converts a sequence of values into +1/-1 indicating whether the next value
increases or decreases. Used for hit rate calculation.

# Arguments
- `y` - Time series vector

# Returns
- Vector of length(y)-1 with values +1 (increase) or -1 (decrease)
"""
function wave(y)
    L=length(y)
    w=zeros(L-1)
    for i in 1:L-1
        w[i]=sign(y[i+1]-y[i])  # +1 if increasing, -1 if decreasing
    end
    return w
end

"""
    hitrate(x, y)

Calculate the hit rate (directional accuracy) between predictions and actuals.

Hit rate measures how often the model correctly predicts the direction of
change (up or down), which is important for trading decisions.

# Arguments
- `x` - Predicted values
- `y` - Actual values

# Returns
- Hit rate as a fraction (0 to 1), where 1 means perfect directional accuracy

# Method
1. Prepend initial value to both series
2. Compute directional changes (waves)
3. Calculate fraction of matching directions
"""
function hitrate(x,y)
    L=length(x)
    # Prepend reference value
    x=vcat(-0.5704088242386152,x)
    y=vcat(-0.5704088242386152,y)
    # Get directional changes
    wx=wave(x)
    wy=wave(y)
    # Return fraction of correct directions
    return sum(wx.==wy)/L
end


# ============================================================================
# Quantum Circuit Callable Extensions
# ============================================================================
# These functions make QCircuit objects callable with different inputs

"""
    (c::QCircuit)(p::Vector)

Make quantum circuit callable with parameter vector.
Sets the circuit parameters and returns the updated circuit.
"""
function (c::QCircuit)(p::Vector)
    return reset_parameters!(c,p)
end

"""
    (c::QCircuit)(ρ::DensityMatrix)

Apply quantum circuit to a density matrix (CPU version).
Performs the quantum operation: ρ → c*ρ*c†
"""
function (c::QCircuit)(ρ::DensityMatrix)
    return c*ρ
end

"""
    (c::QCircuit)(ρ::CuDensityMatrixBatch)

Apply quantum circuit to a batched density matrix (GPU version).
GPU-accelerated circuit application for batch processing.
"""
function (c::QCircuit)(ρ::CuDensityMatrixBatch)
    return c*ρ
end

# ============================================================================
# Example Usage and Visualization Code (Commented Out)
# ============================================================================
# The following commented code shows example usage for printing evaluation metrics
# and creating animated visualizations of predictions vs actual values.

# Example: Print evaluation metrics
# println("Virtual node: 0")
# println("Train_interval:$(wi)")
# println("Hit rate:$(hitrate(P1,y_test))")
# println("MAPE: $(MAPE(P1,y_test))")
# println("MAE: $(MAE(P1,y_test))")
# println("MSE: $(MSE(P1,y_test))")
# println("RMSE: $(RMSE(P1,y_test))")

# Example: Create animated plot showing predictions vs targets for different virtual nodes
# Requires Plots.jl package with @animate macro
# anim = @animate for i in 1:10
#     plot(Vector(K+1:60),[(Ws[i]*signals[i])'[K+1:end],RV[(K+1):60]],label=["Predict" "Target"],lw=3,legendfontsize=18,titlefontsize=24,guidefontsize=24,tickfontsize=18,bottom_margin=5mm,left_margin = 5mm,size=(1000,600),ylims=(-1,-0.3),legend=:bottomleft,xlabel="Time", ylabel="Rv")
#     quiver!([20,30], [-0.45,-0.45], quiver=([-20,20], [0,0]),lw=3)
#     quiver!([56,62], [-0.45,-0.45], quiver=([-4,5], [0,0]),lw=3)
#     vline!([48],lw=2, label= false)
#     annotate!(25,-0.45,("Train",24))
#     annotate!(59,-0.45,("Test",24))
#     annotate!(30,-0.95,("Virtual node=$i",24))
# end

# Example: Static plot with train/test split visualization
# i=10
# plot(Vector(K+1:Ti+12),[(Ws[i]*signals[i])'[K+1:end],vcat(y_train,y_test)],label=["Predict" "Target"],lw=3,legendfontsize=18,titlefontsize=24,guidefontsize=24,tickfontsize=18,bottom_margin=5mm,left_margin = 5mm,size=(1000,600),ylims=(-1,-0.3),legend=:bottomleft,xlabel="Time", ylabel="Rv")
# quiver!([20,30], [-0.45,-0.45], quiver=([-20,20], [0,0]),lw=3)
# quiver!([56,62], [-0.45,-0.45], quiver=([-4,5], [0,0]),lw=3)
# vline!([51],lw=2, label= false)
# annotate!(25,-0.45,("Train",24))
# annotate!(59,-0.45,("Test",24))
# annotate!(30,-0.95,("Virtual node=$i",24))


# ============================================================================
# Data Manipulation Functions
# ============================================================================

"""
    shift(V::Vector{T}, step::Int) where T

Shift a vector forward by 'step' positions, padding with zeros at the beginning.

# Arguments
- `V::Vector{T}` - Input vector to shift
- `step::Int` - Number of positions to shift forward

# Returns
- Shifted vector with zeros in first 'step' positions

# Example
shift([1,2,3,4], 2) → [0, 0, 1, 2]
"""
function shift(V::Vector{T},step::Int) where T
    V1=zeros(T, length(V))
    V1[step+1:end] = V[1:end-step]
    return V1
end

"""
    shift(V::Matrix{T}, step::Int) where T

Shift a matrix forward by 'step' rows, padding with zeros at the beginning.

# Arguments
- `V::Matrix{T}` - Input matrix to shift
- `step::Int` - Number of rows to shift forward

# Returns
- Shifted matrix with zero rows at the top
"""
function shift(V::Matrix{T},step::Int) where T
    V1=zeros(T, size(V))
    V1[step+1:end,:] = V[1:end-step,:]
    return V1
end

"""
    rolling(V::Vector{T}, window::Int) where T

Create a rolling window matrix from a vector.

Constructs a matrix where each column contains the vector values
at different time lags, useful for creating time-delay embeddings.

# Arguments
- `V::Vector{T}` - Input time series vector
- `window::Int` - Window size (number of lags)

# Returns
- Matrix of size (length(V) × window) where column i contains V shifted by i-1

# Example
rolling([1,2,3,4], 2) → [[1,1], [2,1], [3,2], [4,3]]

This is commonly used in reservoir computing to create input with memory.
"""
function rolling(V::Vector{T},window::Int) where T
    M = zeros(T,length(V),window)
    for i in 1:window
        M[i:end,i]=V[1:end-i+1]
    end
    return M
end

# ============================================================================
# Alternative Quantum Reservoir Implementation
# ============================================================================

"""
    Quantum_Reservoir_util(Data, features, QR, Observable, K_delay, VirtualNode, τ, nqubit)

Utility version of quantum reservoir that processes features in a different format.

This variant expects the input data to be pre-arranged with time-delayed features
concatenated into columns, rather than selecting features by name.

# Differences from Quantum_Reservoir:
- Uses pre-arranged feature columns: (k-1)*InputSize+1 to k*InputSize
- Processes data in a specific column format for batch operations
- Otherwise follows the same quantum reservoir computing approach

# Arguments
Same as Quantum_Reservoir, but expects Data in different format:
- Data should have columns organized as: [t-K features, t-K+1 features, ..., t features]

# Returns
- Output matrix (N*VirtualNode × L) of reservoir states
"""
function Quantum_Reservoir_util(Data, features, QR, Observable, K_delay, VirtualNode, τ, nqubit)
    N = length(Observable)
    L = size(Data,1)
    
    InputSize = length(features)

    Output = zeros(N*VirtualNode,L)
    
    δτ = τ/VirtualNode

    δU = CuArray(exp(-im*δτ*Matrix(matrix(QR))))
    U = CuArray(exp(-im*τ*Matrix(matrix(QR))))

    for l in (K_delay+1):L
        cir = QCircuit()
        for i in 1:InputSize
            push!(cir,RyGate(i,rand(),isparas=true))
        end
        ρᵣ = CuDensityMatrixBatch{ComplexF32}(nqubit-InputSize,1)
        for k in K_delay:-1:1

            ρ₁ = CuDensityMatrixBatch{ComplexF32}(InputSize,1)
            # Extract features from pre-arranged columns
            para = Vector(Data[l-1,(k-1)*InputSize+1:k*InputSize])
            cir(para.*pi)
            ρ = ρᵣ⊗(cir(ρ₁))
            if k!=1
                ρ = U*ρ*U'
                ρᵣ=partial_tr(ρ, Vector(1:InputSize))
            else
                it=1
                for v in 1:VirtualNode
                    ρ = δU*ρ*δU'
                    for n in 1:N 
                        Output[it,l] = vec(real(expectation(B[n],ρ)))[1]
                        it+=1
                    end
                end
            end
        end
    end
    return Output
end


# ============================================================================
# Single Reservoir Instance (CPU Version)
# ============================================================================

"""
    Quantum_Reservoir_single(Input, U, U1, Observable, VirtualNode, nqubit, bias)

Process a single time series sample through the quantum reservoir (CPU version).

This is a non-batched, CPU-only version for processing individual samples.
Unlike the main Quantum_Reservoir function, this operates on DensityMatrix
instead of CuDensityMatrixBatch.

# Arguments
- `Input` - Input matrix (K × InputSize) containing K time steps of features
- `U` - Full evolution operator exp(-iτH)
- `U1` - Small step evolution operator exp(-iδτH) for virtual nodes
- `Observable` - Vector of observables to measure
- `VirtualNode::Int` - Number of virtual nodes
- `nqubit::Int` - Total number of qubits
- `bias` - Bias parameters for initializing reservoir state

# Returns
- Output vector (N*VirtualNode) of reservoir measurements

# Use Case
Useful for single-sample predictions or when GPU is not available.
Can be parallelized across samples using threading (see commented code at end).
"""
function Quantum_Reservoir_single(Input, U, U1, Observable, VirtualNode, nqubit,bias)
    N = length(Observable)
    K, InputSize = size(Input)

    # Create input encoding circuit
    cir = QCircuit()
    for i in 1:InputSize
        push!(cir,RyGate(i,rand(),isparas=true))
    end

    # Initialize reservoir state with bias
    cir2 = QCircuit()
    for i in 1:nqubit-InputSize
        push!(cir2,RyGate(i,rand(),isparas=true))
    end
    cir2(bias.*pi)
    ρᵣ = DensityMatrix(nqubit-InputSize)
    ρᵣ=cir2(ρᵣ)
    
    Output = zeros(N*VirtualNode)

    # Process each time step
    for k in 1:K
        ρ₁ = DensityMatrix(InputSize)
        cir(Input[k,:].*pi)
        ρ = ρᵣ⊗(cir(ρ₁))
        
        if k!=K
            # Not final step: evolve and trace out input
            ρ = U*ρ*U'
            ρᵣ=partial_tr(ρ, Vector(1:InputSize))
        else
            # Final step: use virtual nodes
            it=1
            for v in 1:VirtualNode
                ρ = U1*ρ*U1'
                for n in 1:N 
                    Output[it] = real(expectation(Observable[n],ρ))
                    it+=1
                end
            end
        end
    end
    return Output
end

# ============================================================================
# Alternative Threaded Implementation (Commented Out)
# ============================================================================
# This is an alternative implementation using multi-threading for parallelization
# across time steps instead of GPU acceleration. Each thread processes a different
# time step using the CPU-based Quantum_Reservoir_single function.
#
# Advantages:
# - No GPU required
# - Can be more efficient for smaller problems
# - Better for systems with many CPU cores
#
# Disadvantages:
# - Slower than GPU implementation for large-scale problems
# - Higher memory usage due to per-thread state
#
# Usage: Uncomment and replace the main Quantum_Reservoir function to use threading.

# function Quantum_Reservoir(Data, features, QR, Observable, K_delay, VirtualNode, τ, nqubit)
#     N = length(Observable)
#     L = size(Data,1)
#     x_data = zeros(N*VirtualNode,L)
#     Input = Matrix(Data[:,features])
#     δv = τ/VirtualNode
#     U = exp(-im*τ*Matrix(matrix(QR)))
#     U1 = exp(-im*δv*Matrix(matrix(QR)))
#     # Use multi-threading to process each time step in parallel
#     Threads.@threads for l in K_delay+1:L
#         x_data[:,l] = Quantum_Reservoir_single(Input[l-K_delay:l,:], U, U1, Observable, VirtualNode, nqubit)
#     end
#     return x_data
# end