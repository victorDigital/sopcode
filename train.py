import numpy as np
from mnist_dataloader import MnistDataloader  # Added import statement
from activations import phi, dphi, dphiofphi, softmax

# Load MNIST data
print("Loading MNIST data...")
mnist_dataloader = MnistDataloader(
        'mnist/train-images.idx3-ubyte', 'mnist/train-labels.idx1-ubyte',
        'mnist/t10k-images.idx3-ubyte', 'mnist/t10k-labels.idx1-ubyte'
    )
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
print("MNIST data loaded.")

x_train = np.array(x_train).reshape(-1, 28*28) / 255.0
y_train_one_hot = np.zeros((len(y_train), 10))
y_train_one_hot[np.arange(len(y_train)), y_train] = 1

x_test = np.array(x_test).reshape(-1, 28*28) / 255.0
y_test_one_hot = np.zeros((len(y_test), 10))
y_test_one_hot[np.arange(len(y_test)), y_test] = 1

class NeuralNetwork:
    def __init__(self):
        self.noNodes = [16, 16]
        self.nI = x_train.shape[1]
        self.nO = y_train_one_hot.shape[1]
        self.W1 = np.random.uniform(-1, 1, (self.nI, self.noNodes[0]))
        self.b1 = np.random.uniform(-1, 1, self.noNodes[0])
        self.W2 = np.random.uniform(-1, 1, (self.noNodes[0], self.noNodes[1]))
        self.b2 = np.random.uniform(-1, 1, self.noNodes[1])
        self.W3 = np.random.uniform(-1, 1, (self.noNodes[1], self.nO))
        self.b3 = np.random.uniform(-1, 1, self.nO)

    def predict(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = phi(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = phi(Z2)
        Z3 = A2 @ self.W3 + self.b3
        A3 = softmax(Z3)
        return {'A1': A1, 'A2': A2, 'A3': A3, 'Z1': Z1, 'Z2': Z2, 'Z3': Z3}

    def update(self, X, Y, pred, rate=1):
        m = X.shape[0]
        A1, A2, A3 = pred['A1'], pred['A2'], pred['A3']
        dZ3 = A3 - Y
        dW3 = (A2.T @ dZ3) / m
        db3 = np.sum(dZ3, axis=0) / m
        dZ2 = (dZ3 @ self.W3.T) * dphiofphi(A2)
        dW2 = (A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0) / m
        dZ1 = (dZ2 @ self.W2.T) * dphiofphi(A1)
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0) / m
        self.W1 -= rate * dW1
        self.b1 -= rate * db1
        self.W2 -= rate * dW2
        self.b2 -= rate * db2
        self.W3 -= rate * dW3
        self.b3 -= rate * db3

    def train(self, X, Y, epochs):
        # randomize the data
        idx = np.random.permutation(X.shape[0])
        X, Y = X[idx], Y[idx]
        
        # make training batches of 10000 samples
        X_batches = np.array_split(X, X.shape[0] // 10000)
        Y_batches = np.array_split(Y, Y.shape[0] // 10000)
        for epoch in range(1, epochs + 1):
            for X_batch, Y_batch in zip(X_batches, Y_batches):
                pred = self.predict(X_batch)
                self.update(X_batch, Y_batch, pred, rate=1)
            if epoch % 10 == 0:
                loss = -np.mean(np.sum(Y_batch * np.log(pred['A3'] + 1e-8), axis=1))
                accuracy = np.mean(np.argmax(pred['A3'], axis=1) == np.argmax(Y_batch, axis=1))
                # report the accuracy on the test set
                test_pred = self.predict(x_test)
                test_accuracy = np.mean(np.argmax(test_pred['A3'], axis=1) == y_test)
                print(f"Epoch {epoch} Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%, Test Accuracy: {test_accuracy * 100:.2f}%")
            if epoch % 100 == 0:
                self.export_to_file("model.npz")
    
    def export_to_file(self, filename):
        np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)

nn = NeuralNetwork()
nn.train(x_train, y_train_one_hot, epochs=10000)