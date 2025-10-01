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



# Dados fictícios: [umidade, pressão, temperatura, nuvens, vento]
X = np.array([
    [85, 1005, 28, 8, 12],   # choveu
    [40, 1018, 32, 2, 5],    # não choveu
    [78, 1007, 30, 7, 15],   # choveu
    [55, 1020, 33, 3, 6]     # não choveu
])

# Saída: 1 = Chove, 0 = Não chove
y = np.array([1, 0, 1, 0])

# Treinamento
p = Perceptron(lr=0.01, epochs=50)
p.fit(X, y)

# Testando previsão
teste = np.array([0, 0, 0, 0, 0])  # condições novas
print(p.predict(teste))
