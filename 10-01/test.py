from sklearn.metrics import accuracy_score
import numpy as np

class Perceptron:
    def __init__(self, lr=0.1, epochs=20):
        self.lr = lr
        self.epochs = epochs
    
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1] + 1)  # +1 para bias
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                update = self.lr * (target - self.predict_raw(xi))
                self.w[1:] += update * xi
                self.w[0] += update  # bias
    
    def net_input(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]
    
    def predict_raw(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    
    def predict(self, X):
        resultado = self.predict_raw(X)
        return "Chove" if resultado == 1 else "Não chove"
    
# Dados fictícios
X = np.array([
    [85, 1005, 28, 8, 12],
    [40, 1018, 32, 2, 5],
    [78, 1007, 30, 7, 15],
    [55, 1020, 33, 3, 6],
    [88, 1006, 27, 9, 11]
])
y = np.array([1, 0, 1, 0, 1])  # 1 = Chove, 0 = Não chove

# Testando com diferentes parâmetros
configs = [
    {"epochs": 5, "lr": 0.5},
    {"epochs": 20, "lr": 0.1},
    {"epochs": 100, "lr": 0.01}
]

for cfg in configs:
    p = Perceptron(lr=cfg["lr"], epochs=cfg["epochs"])
    p.fit(X, y)
    y_pred = [p.predict_raw(xi) for xi in X]
    acc = accuracy_score(y, y_pred)
    print(f"Épocas={cfg['epochs']} | LR={cfg['lr']} -> Acurácia: {acc:.2f}")
