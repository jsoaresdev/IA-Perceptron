
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

# Etapa 1: Configuração inicial
os.makedirs("output", exist_ok=True)

# Etapa 2: Geração dos dados sintéticos
mean1, cov1 = [0, 0], [[1, 1], [1, 3]]
mean2, cov2 = [0, 10], [[1, 1], [1, 3]]
n_samples = 400

X = pd.DataFrame(
    np.vstack([
        np.random.multivariate_normal(mean1, cov1, size=n_samples // 2),
        np.random.multivariate_normal(mean2, cov2, size=n_samples // 2)
    ]),
    columns=["x1", "x2"]
)
y = pd.Series([0] * (n_samples // 2) + [1] * (n_samples // 2), name="target")

# Etapa 3: Visualização dos dados
plt.figure()
plt.scatter(X["x1"], X["x2"], c=y, cmap="bwr", edgecolor="k")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Scatterplot dos dados")
plt.savefig("output/scatterplot.png")
plt.close()

# Etapa 4: Classe Perceptron
class Perceptron:
    def __init__(self, learning_rate=0.001, n_epochs=30, random_state=42):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.errors_ = []

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.weights_ = rng.normal(0.0, 0.01, X.shape[1])
        self.bias_ = 0.0
        self.errors_ = []

        for _ in range(self.n_epochs):
            errors = 0
            for xi, target in zip(X.values, y):
                prediction = self.predict(xi)
                update = self.lr * (target - prediction)
                self.weights_ += update * xi
                self.bias_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            if errors == 0:
                break
        return self

    def net_input(self, X):
        return np.dot(X, self.weights_) + self.bias_

    def predict(self, X):
        return np.where(self.net_input(X) > 0.0, 1, 0)

# Etapa 5: Curva de convergência
model = Perceptron()
model.fit(X, y)

plt.figure()
plt.plot(range(1, len(model.errors_) + 1), model.errors_, marker="o")
plt.xlabel("Época")
plt.ylabel("Erros de classificação")
plt.title("Curva de convergência")
plt.savefig("output/convergence.png")
plt.close()

# Etapa 6: Fronteira de decisão
def plot_decision_boundary(model, X, y, title, filename):
    x_min, x_max = X["x1"].min() - 1, X["x1"].max() + 1
    y_min, y_max = X["x2"].min() - 1, X["x2"].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="bwr")
    plt.scatter(X["x1"], X["x2"], c=y, cmap="bwr", edgecolor="k")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.savefig(f"output/{filename}")
    plt.close()

plot_decision_boundary(model, X, y, "Fronteira de Decisão", "decision_boundary.png")

# Etapa 7: Evolução da fronteira
plt.figure()
for epoch in range(1, 31):
    temp_model = Perceptron(n_epochs=epoch)
    temp_model.fit(X, y)
    
    # Cálculo da linha de decisão
    if temp_model.weights_[1] != 0:
        slope = -temp_model.weights_[0] / temp_model.weights_[1]
        intercept = -temp_model.bias_ / temp_model.weights_[1]
        x_vals = np.array([X["x1"].min() - 1, X["x1"].max() + 1])
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, label=f"Época {epoch}" if epoch in [1, 10, 20, 30] else "", alpha=0.3)

plt.scatter(X["x1"], X["x2"], c=y, cmap="bwr", edgecolor="k")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Evolução da Fronteira de Decisão")
plt.legend()
plt.savefig("output/decision_boundary_evolution.png")
plt.close()
