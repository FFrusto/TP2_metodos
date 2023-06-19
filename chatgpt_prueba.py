import numpy as np

# Función de activación sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Función de pérdida o función objetivo
def loss_function(y_real, y_predicted):
    return 0.5 * np.sum((y_real - y_predicted) ** 2)

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(hidden_dim, input_dim)
        self.b1 = np.random.randn(hidden_dim, 1)
        self.W2 = np.random.randn(output_dim, hidden_dim)
        self.b2 = np.random.randn(output_dim)

    def forward(self, x):
        self.z1 = np.matmul(self.W1, x) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.matmul(self.W2, self.a1) + self.b2
        return self.z2

    def numerical_gradient(self, x, y_real):
        eps = 1e-4
        grads = {}
        for param_name, param_value in zip(["W1", "b1", "W2", "b2"], [self.W1, self.b1, self.W2, self.b2]):
            grad = np.zeros_like(param_value)
            it = np.nditer(param_value, flags=['multi_index'])
            while not it.finished:
                original_value = param_value[it.multi_index]
                param_value[it.multi_index] = original_value + eps
                y_plus = self.forward(x)
                loss_plus = loss_function(y_real, y_plus)
                param_value[it.multi_index] = original_value - eps
                y_minus = self.forward(x)
                loss_minus = loss_function(y_real, y_minus)
                grad[it.multi_index] = (loss_plus - loss_minus) / (2 * eps)
                param_value[it.multi_index] = original_value
                it.iternext()
            grads[param_name] = grad
        return grads

    def fit(self, X_train, y_train, learning_rate, epochs):
        for i in range(epochs):
            grads = self.numerical_gradient(X_train, y_train)
            for param_name in ["W1", "b1", "W2", "b2"]:
                exec(f"self.{param_name} -= learning_rate * grads[param_name]")
            if i % 100 == 0:
                y_pred = self.forward(X_train)
                loss = loss_function(y_train, y_pred)
                print(f'Epoch {i}, loss: {loss}')


import matplotlib.pyplot as plt

# Configuraciones de red
input_dim = 10
hidden_dim = 5
output_dim = 1

# Configuraciones de entrenamiento
learning_rate = 1e-3
epochs = 1000

# Crear la red
network = NeuralNetwork(input_dim, hidden_dim, output_dim)

# Crear los datos
X_train = np.random.randn(input_dim, 315)
y_train = np.random.randn(output_dim, 315)
X_test = np.random.randn(input_dim, 99)
y_test = np.random.randn(output_dim, 99)

# Listas para guardar los resultados
train_losses = []
test_losses = []

# Entrenamiento
for i in range(epochs):
    # Ajuste y cálculo de la pérdida de entrenamiento
    network.fit(X_train, y_train, learning_rate,1)
    train_loss = loss_function(y_train, network.forward(X_train))
    train_losses.append(train_loss)

    # Cálculo de la pérdida de prueba
    test_loss = loss_function(y_test, network.forward(X_test))
    test_losses.append(test_loss)

    # Mostrar el progreso
    if i % 100 == 0:
        print(f'Epoch {i}, train loss: {train_loss}, test loss: {test_loss}')

# Graficar las pérdidas
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train loss')
plt.plot(test_losses, label='Test loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
