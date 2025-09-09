import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

data = {
    'AND': (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 0, 1])),
    'OR':  (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 1])),
    'XOR': (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 0])),
}

for gate, (X, y) in data.items():
    print(f"\n--- {gate} Gate ---")

    # Multi-layer perceptron (can solve XOR)
    mlp = MLPClassifier(hidden_layer_sizes=(3, 2), activation='tanh',
                        solver='lbfgs', max_iter=2000, random_state=42)
    mlp.fit(X, y)

    y_pred = mlp.predict(X)
    acc = accuracy_score(y, y_pred) * 100
    print(f"MLP accuracy: {acc:.2f}%")
    print(f"MLP Predictions: {y_pred}")
    print(f"True Labels: {y}")

    # Single-layer perceptron (works only for linearly separable problems)
    perceptron = Perceptron(max_iter=100, eta0=1, random_state=42)
    perceptron.fit(X, y)

    y_pred = perceptron.predict(X)
    acc = accuracy_score(y, y_pred) * 100
    print(f"Perceptron accuracy: {acc:.2f}%")
    print(f"Perceptron Predictions: {y_pred}")
    print(f"True Labels: {y}")
