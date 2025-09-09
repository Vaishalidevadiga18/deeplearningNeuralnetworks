import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0)

# Input values
x = np.linspace(-10, 10, 400)

# Compute outputs
y = {
    'Sigmoid': sigmoid(x),
    'Tanh': tanh(x),
    'ReLU': relu(x),
    'Softmax': softmax(np.array([x, x*0.5, x*0.2])).T
}

# Plot
plt.figure(figsize=(12, 8))
for i, (name, values) in enumerate(y.items()):
    plt.subplot(2, 2, i+1)
    if name == 'Softmax':
        for j in range(values.shape[1]):
            plt.plot(x, values[:, j], label=f"Set {j+1}")
    else:
        plt.plot(x, values, label=name)
    plt.title(f"{name} Activation Function")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
