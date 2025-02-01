import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=1000):
        self.weights = np.random.randn(input_size + 1) * 0.01  # Including bias term
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        ### START CODE HERE ###
        ### A função de ativação verifica se cada valor de x é maior ou igual a zero.
        ### Caso seja, retorna 1; caso contrário, retorna -1.
        ### Como a entrada pode ser um vetor, utilizamos np.where,
        ### que aplica essa verificação elemento por elemento.
        ### Exemplo: np.where(np.array([2, 0, 2, 3, -3]) >= 0, 1, -1) -> [1, 1, 1, 1, -1]
        return np.where(x>=0,1,-1)
        ### END CODE HERE ###
        
    def predict(self, X):
        ### START CODE HERE ###
        ### Adiciona o termo de bias à matriz de entrada X.
        ### O bias é representado por uma coluna adicional de 1s.
        ### Exemplo: Para X = [[2], [0], [2], [3], [-3]], o resultado será:
        ### [[2, 1], [0, 1], [2, 1], [3, 1], [-3, 1]]
        x_bias=np.c_[X,np.ones(X.shape[0])]
        ### Calcula a soma ponderada dos pesos com as entradas (produto escalar).
        ### A equação é equivalente à função linear: soma_ponderada = XW + b
        ### Onde:
        ### - `x_bias` contém os valores de entrada e o bias.
        ### - `self.weights` contém os pesos do modelo.
        ### Exemplo:
        ### Se x_bias = [[2, 1], [0, 1], [2, 1], [3, 1], [-3, 1]] e
        ### self.weights = [0.1, 0.2], então:
        ### soma_ponderada = np.dot(x_bias, self.weights) = [0.4, 0.2, 0.4, 0.5, -0.1]
        soma_ponderada = np.dot(x_bias, self.weights) ### Realiza a operação da função afim (xW+b)
        ### Aplica a função de ativação sobre a soma ponderada.
        ### A ativação converte os valores contínuos para -1 ou 1, 
        ### representando as classes do Perceptron.
        return self.activation(soma_ponderada) 
        ### Retorna o resultado da soma_ponderada depois de passar pelo função de 
        ### ativação para normalizar os valores para o que esperamos (-1,1)
        ### END CODE HERE ###

    def fit(self, X, y):
        ### START CODE HERE ###
        ### Função responsavel pelo treinamento do perceptron para achar o melhor conjunto de pesos (valores para b e para w)
        ### regra de aprendizado do Perceptron: w = w + n (y - y_previsto)X
        ### TODO
        x_bias=np.c_[X,np.ones(X.shape[0])]
        for _ in range (self.epochs):
            for xi, target in zip(x_bias,y):
                prediction = self.predict(xi[:-1].reshape(1, -1))  # Remove o bias antes de chamar predict
                if prediction != target: # se tiver errado, ajusta o bias
                    # Atualizar os pesos sem afetar o bias diretamente
                    self.weights[:-1] += self.learning_rate * (target - prediction) * xi[:-1]
                    # Atualizar o bias separadamente
                    self.weights[-1] += self.learning_rate * (target - prediction)
        ### END CODE HERE ###

def generate_data(seed, samples, noise):
    """
    Generates two-class classification data with overlapping clusters.
    """
    np.random.seed(seed)
    X, y = make_blobs(n_samples=samples, centers=2, cluster_std=noise, random_state=seed)
    y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1 for perceptron
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Train a single-layer perceptron using NumPy.")
    parser.add_argument('--registration_number', type=int, required=True, help="Student's registration number (used as seed)")
    parser.add_argument('--samples', type=int, default=200, help="Number of data samples")
    parser.add_argument('--noise', type=float, default=1.5, help="Standard deviation of clusters")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Perceptron learning rate")
    parser.add_argument('--epochs', type=int, default=1000, help="Number of training epochs")
    
    args = parser.parse_args()

    # Generate data
    X, y = generate_data(args.registration_number, args.samples, args.noise)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.registration_number)

    # Train the perceptron
    perceptron = Perceptron(input_size=2, learning_rate=args.learning_rate, epochs=args.epochs)
    perceptron.fit(X_train, y_train)

    # Evaluate the perceptron
    y_pred = perceptron.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    print(f"Perceptron Training Completed.")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Visualizing decision boundary
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolors="k", alpha=0.6)
    
    # Plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = perceptron.predict(grid_points).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.title(f"Perceptron Decision Boundary (Accuracy: {accuracy:.4f})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

if __name__ == "__main__":
    main()