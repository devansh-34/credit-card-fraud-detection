
import numpy as np

print("=" * 60)
print("       NEURAL NETWORK FUNDAMENTALS")
print("=" * 60)

# ==================================================
# PART 1: TENSOR MATHEMATICS
# ==================================================

print("\nPART 1: TENSOR MATHEMATICS")

# 1. Create vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("\nVector A:", a)
print("Vector B:", b)

# 2. Vector addition
print("\nVector Addition:", a + b)

# 3. Element-wise multiplication
print("Element-wise Multiplication:", a * b)

# 4. Inner product / Dot product
print("Dot Product:", np.dot(a, b))

# 5. Create a matrix
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("\nMatrix:")
print(matrix)

# 6. Matrix transpose
print("\nMatrix Transpose:")
print(matrix.T)

# 7. Reshaping
reshaped = np.arange(1, 7).reshape(3, 2)

print("\nReshaped Matrix:")
print(reshaped)

# 8. Broadcasting
x = np.array([[1, 2, 3],
              [4, 5, 6]])

print("\nBroadcasting (Matrix + 10):")
print(x + 10)

# 9. Singular Value Decomposition
print("\nSINGULAR VALUE DECOMPOSITION (SVD)")

U, S, VT = np.linalg.svd(matrix, full_matrices=False)

print("U:")
print(U)

print("Singular values:")
print(S)

print("VT:")
print(VT)

# ==================================================
# PART 2: MCCULLOCH-PITTS NEURON
# ==================================================

print("\nPART 2: MCCULLOCH-PITTS NEURON")

def mp_neuron(inputs, weights, threshold):
    weighted_sum = np.dot(inputs, weights)

    if weighted_sum >= threshold:
        return 1
    else:
        return 0

weights = np.array([1, 1])
threshold = 2

print("\nAND GATE OUTPUTS")

for inputs in [
    np.array([0, 0]),
    np.array([0, 1]),
    np.array([1, 0]),
    np.array([1, 1])
]:
    output = mp_neuron(inputs, weights, threshold)
    print(inputs, "=", output)

# ==================================================
# PART 3: SIGMOID NEURON
# ==================================================

print("\nPART 3: SIGMOID NEURON")

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

inputs = np.array([0.5, 0.8])

weights = np.array([0.4, 0.6])
bias = -0.2

z = np.dot(inputs, weights) + bias
output = sigmoid(z)

print("Weighted sum:", round(z, 4))
print("Sigmoid output:", round(output, 4))

# ==================================================
# PART 4: ACTIVATION FUNCTIONS
# ==================================================

print("\nPART 4: ACTIVATION FUNCTIONS")

def step_function(x):
    return np.where(x >= 0, 1, 0)

def tanh_function(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

values = np.array([-2, -1, 0, 1, 2])

print("Input:", values)
print("Step:", step_function(values))
print("Sigmoid:", np.round(sigmoid(values), 3))
print("Tanh:", np.round(tanh_function(values), 3))
print("ReLU:", relu(values))

# ==================================================
# FINAL
# ==================================================

print("\n" + "=" * 60)
print("       NEURAL FUNDAMENTALS COMPLETED!")
print("=" * 60)
