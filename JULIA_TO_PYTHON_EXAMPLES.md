# Julia to Python Code Examples

This document provides side-by-side comparisons of Julia and Python code for common operations in the Quantum Reservoir Computing project.

## Basic Operations

### Data Normalization

**Julia:**
```julia
x = [1.0, 2.0, 3.0, 4.0, 5.0]
normalized = normalization(x, -1, 1)
```

**Python:**
```python
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
normalized = normalization(x, -1, 1)
```

### Denormalization

**Julia:**
```julia
x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [-1.0, -0.5, 0.0, 0.5, 1.0]
original = denormalization(x, y, -1, 1)
```

**Python:**
```python
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
original = denormalization(x, y, -1, 1)
```

## Evaluation Metrics

### Computing Multiple Metrics

**Julia:**
```julia
predictions = [0.1, 0.2, 0.15, 0.25, 0.3]
actuals = [0.12, 0.18, 0.16, 0.24, 0.28]

mse_val = MSE(predictions, actuals)
rmse_val = RMSE(predictions, actuals)
mae_val = MAE(predictions, actuals)
mape_val = MAPE(predictions, actuals)
hit = hitrate(predictions, actuals)

println("MSE: $mse_val")
println("RMSE: $rmse_val")
println("MAE: $mae_val")
println("MAPE: $mape_val%")
println("Hit Rate: $hit")
```

**Python:**
```python
predictions = np.array([0.1, 0.2, 0.15, 0.25, 0.3])
actuals = np.array([0.12, 0.18, 0.16, 0.24, 0.28])

mse_val = MSE(predictions, actuals)
rmse_val = RMSE(predictions, actuals)
mae_val = MAE(predictions, actuals)
mape_val = MAPE(predictions, actuals)
hit = hitrate(predictions, actuals)

print(f"MSE: {mse_val}")
print(f"RMSE: {rmse_val}")
print(f"MAE: {mae_val}")
print(f"MAPE: {mape_val}%")
print(f"Hit Rate: {hit}")
```

## Utility Functions

### Generating Coupling Matrix

**Julia:**
```julia
nqubit = 4
J = 1.0
ps = coeff_matrix(nqubit, J)
println("Coupling matrix:")
println(ps)
println("Max eigenvalue: $(max(eigvals(ps)...))")
```

**Python:**
```python
nqubit = 4
J = 1.0
ps = coeff_matrix(nqubit, J)
print("Coupling matrix:")
print(ps)
print(f"Max eigenvalue: {np.max(eigvals(ps).real)}")
```

### Computing Wave (Directional Changes)

**Julia:**
```julia
y = [1.0, 2.0, 1.5, 3.0, 2.5]
directions = wave(y)
println("Time series: $y")
println("Directions: $directions")
```

**Python:**
```python
y = np.array([1.0, 2.0, 1.5, 3.0, 2.5])
directions = wave(y)
print(f"Time series: {y}")
print(f"Directions: {directions}")
```

### Hit Rate Calculation

**Julia:**
```julia
predictions = [0.1, 0.2, 0.15, 0.25, 0.3]
actuals = [0.12, 0.18, 0.16, 0.24, 0.28]
hit = hitrate(predictions, actuals)
println("Hit rate: $hit")
```

**Python:**
```python
predictions = np.array([0.1, 0.2, 0.15, 0.25, 0.3])
actuals = np.array([0.12, 0.18, 0.16, 0.24, 0.28])
hit = hitrate(predictions, actuals)
print(f"Hit rate: {hit}")
```

## Data Manipulation

### Shifting Arrays

**Julia:**
```julia
V = [1, 2, 3, 4, 5]
V_shifted = shift(V, 2)
println("Original: $V")
println("Shifted: $V_shifted")
```

**Python:**
```python
V = np.array([1, 2, 3, 4, 5])
V_shifted = shift(V, 2)
print(f"Original: {V}")
print(f"Shifted: {V_shifted}")
```

### Rolling Window

**Julia:**
```julia
V = [1, 2, 3, 4, 5]
window = 3
M = rolling(V, window)
println("Original: $V")
println("Rolling window (size=$window):")
println(M)
```

**Python:**
```python
V = np.array([1, 2, 3, 4, 5])
window = 3
M = rolling(V, window)
print(f"Original: {V}")
print(f"Rolling window (size={window}):")
print(M)
```

## QLIKE Loss

### Computing QLIKE

**Julia:**
```julia
forecasts = [0.1, 0.2, 0.15, 0.25, 0.3]
actuals = [0.12, 0.18, 0.16, 0.24, 0.28]

qlike1 = compute_qlike(forecasts, actuals)
qlike2 = compute_qlike2(forecasts, actuals)

println("QLIKE (version 1): $qlike1")
println("QLIKE (version 2): $qlike2")
```

**Python:**
```python
forecasts = np.array([0.1, 0.2, 0.15, 0.25, 0.3])
actuals = np.array([0.12, 0.18, 0.16, 0.24, 0.28])

qlike1 = compute_qlike(forecasts, actuals)
qlike2 = compute_qlike2(forecasts, actuals)

print(f"QLIKE (version 1): {qlike1}")
print(f"QLIKE (version 2): {qlike2}")
```

## Working with the Model Structure

### Creating and Using MyModel

**Julia:**
```julia
# Create model
model = MyModel(100, 1, ["RV", "MKT", "DP"])

# Access fields
println("Number of features: $(model.L)")
println("Output length: $(model.OutLen)")
println("Features used: $(model.Features)")
println("Weight matrix shape: $(size(model.W))")
```

**Python:**
```python
# Create model
model = MyModel(100, 1, ["RV", "MKT", "DP"])

# Access fields
print(f"Number of features: {model.L}")
print(f"Output length: {model.OutLen}")
print(f"Features used: {model.Features}")
print(f"Weight matrix shape: {model.W.shape}")
```

## Array Operations

### Concatenation

**Julia:**
```julia
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]
c = vcat(a, b)
println("Concatenated: $c")
```

**Python:**
```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
c = np.concatenate([a, b])
print(f"Concatenated: {c}")
```

### Broadcasting

**Julia:**
```julia
x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = x .* 2 .+ 1
println("Original: $x")
println("Transformed: $y")
```

**Python:**
```python
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = x * 2 + 1  # NumPy broadcasting is automatic
print(f"Original: {x}")
print(f"Transformed: {y}")
```

### Element-wise Operations

**Julia:**
```julia
x = [1.0, 2.0, 3.0]
y = [4.0, 5.0, 6.0]
z = x .+ y
println("Sum: $z")

w = x .* y
println("Product: $w")
```

**Python:**
```python
x = np.array([1.0, 2.0, 3.0])
y = np.array([4.0, 5.0, 6.0])
z = x + y  # Automatic element-wise
print(f"Sum: {z}")

w = x * y  # Automatic element-wise
print(f"Product: {w}")
```

## Statistical Operations

### Basic Statistics

**Julia:**
```julia
using Statistics

x = [1.0, 2.0, 3.0, 4.0, 5.0]
println("Mean: $(mean(x))")
println("Std: $(std(x))")
println("Max: $(maximum(x))")
println("Min: $(minimum(x))")
```

**Python:**
```python
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"Mean: {np.mean(x)}")
print(f"Std: {np.std(x)}")
print(f"Max: {np.max(x)}")
print(f"Min: {np.min(x)}")
```

## Matrix Operations

### Matrix Multiplication

**Julia:**
```julia
A = [1.0 2.0; 3.0 4.0]
B = [5.0 6.0; 7.0 8.0]
C = A * B
println("Matrix product:")
println(C)
```

**Python:**
```python
A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[5.0, 6.0], [7.0, 8.0]])
C = A @ B  # or np.dot(A, B)
print("Matrix product:")
print(C)
```

### Kronecker Product (Tensor Product)

**Julia:**
```julia
using LinearAlgebra

A = [1.0 2.0; 3.0 4.0]
B = [5.0 6.0; 7.0 8.0]
C = kron(A, B)
println("Kronecker product:")
println(C)
```

**Python:**
```python
A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[5.0, 6.0], [7.0, 8.0]])
C = np.kron(A, B)
print("Kronecker product:")
print(C)
```

## Indexing Differences

### Array Indexing

**Julia (1-based):**
```julia
arr = [10, 20, 30, 40, 50]
println(arr[1])      # First element: 10
println(arr[end])    # Last element: 50
println(arr[2:4])    # Elements 2 to 4: [20, 30, 40]
```

**Python (0-based):**
```python
arr = np.array([10, 20, 30, 40, 50])
print(arr[0])        # First element: 10
print(arr[-1])       # Last element: 50
print(arr[1:4])      # Elements index 1 to 3: [20, 30, 40]
```

### Matrix Indexing

**Julia (1-based):**
```julia
M = [1 2 3; 4 5 6; 7 8 9]
println(M[1, 1])     # First element: 1
println(M[1, :])     # First row: [1, 2, 3]
println(M[:, 1])     # First column: [1, 4, 7]
```

**Python (0-based):**
```python
M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(M[0, 0])       # First element: 1
print(M[0, :])       # First row: [1, 2, 3]
print(M[:, 0])       # First column: [1, 4, 7]
```

## Complete Example: Processing Time Series Data

**Julia:**
```julia
# Load and normalize data
raw_data = [1.5, 2.3, 1.8, 2.9, 3.1]
normalized_data = normalization(raw_data, -1, 1)

# Create rolling window
window_size = 3
windowed_data = rolling(normalized_data, window_size)

# Generate coupling matrix
nqubit = 4
J = 1.0
coupling = coeff_matrix(nqubit, J)

# Make predictions (example)
predictions = [0.2, 0.3, 0.25]
actuals = [0.22, 0.28, 0.26]

# Evaluate
println("Metrics:")
println("MSE: $(MSE(predictions, actuals))")
println("RMSE: $(RMSE(predictions, actuals))")
println("Hit Rate: $(hitrate(predictions, actuals))")
```

**Python:**
```python
# Load and normalize data
raw_data = np.array([1.5, 2.3, 1.8, 2.9, 3.1])
normalized_data = normalization(raw_data, -1, 1)

# Create rolling window
window_size = 3
windowed_data = rolling(normalized_data, window_size)

# Generate coupling matrix
nqubit = 4
J = 1.0
coupling = coeff_matrix(nqubit, J)

# Make predictions (example)
predictions = np.array([0.2, 0.3, 0.25])
actuals = np.array([0.22, 0.28, 0.26])

# Evaluate
print("Metrics:")
print(f"MSE: {MSE(predictions, actuals)}")
print(f"RMSE: {RMSE(predictions, actuals)}")
print(f"Hit Rate: {hitrate(predictions, actuals)}")
```

## Key Takeaways

1. **Function calls are nearly identical** - Most function names and signatures are the same
2. **Array creation differs** - Use `np.array()` in Python vs `[]` in Julia
3. **Indexing is different** - Python is 0-based, Julia is 1-based
4. **Broadcasting is automatic in NumPy** - No need for `.` operator
5. **Print statements differ** - `println()` in Julia vs `print()` in Python
6. **String interpolation** - `$variable` in Julia vs `f"{variable}"` in Python

The Python conversion maintains the same API and functionality, making migration straightforward!
