import numpy as np
from mnist_dataloader import MnistDataloader  # Added import statement
from activations import sigmoid_activation, sigmoid_derivative, sigmoid_prime, softmax
import time

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
        A1 = sigmoid_activation(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = sigmoid_activation(Z2)
        Z3 = A2 @ self.W3 + self.b3
        A3 = softmax(Z3)
        return {'A1': A1, 'A2': A2, 'A3': A3, 'Z1': Z1, 'Z2': Z2, 'Z3': Z3}

    def update(self, X, Y, pred, rate=1):
        m = X.shape[0]
        A1, A2, A3 = pred['A1'], pred['A2'], pred['A3']
        dZ3 = A3 - Y
        dW3 = (A2.T @ dZ3) / m
        db3 = np.sum(dZ3, axis=0) / m
        dZ2 = (dZ3 @ self.W3.T) * sigmoid_prime(A2)
        dW2 = (A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0) / m
        dZ1 = (dZ2 @ self.W2.T) * sigmoid_prime(A1)
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0) / m
        self.W1 -= rate * dW1
        self.b1 -= rate * db1
        self.W2 -= rate * dW2
        self.b2 -= rate * db2
        self.W3 -= rate * dW3
        self.b3 -= rate * db3
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

    def train(self, X, Y, epochs):
        # randomize the data
        idx = np.random.permutation(X.shape[0])
        X, Y = X[idx], Y[idx]
        
        # make training batches of 10000 samples
        X_batches = np.array_split(X, X.shape[0] // 100)
        Y_batches = np.array_split(Y, Y.shape[0] // 100)
        for epoch in range(1, epochs + 1):
            for X_batch, Y_batch in zip(X_batches, Y_batches):
                pred = self.predict(X_batch)
                self.update(X_batch, Y_batch, pred, rate=0.1)
            if epoch % 10 == 0:
                pred = self.predict(X)
                loss = -np.mean(np.sum(Y * np.log(pred['A3'] + 1e-8), axis=1))
                accuracy = np.mean(np.argmax(pred['A3'], axis=1) == np.argmax(Y, axis=1))
                # report the accuracy on the test set
                test_pred = self.predict(x_test)
                test_accuracy = np.mean(np.argmax(test_pred['A3'], axis=1) == y_test)
                print(f"Epoch {epoch} Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%, Test Accuracy: {test_accuracy * 100:.2f}%")
            if epoch % 100 == 0:
                self.export_to_file("model.npz")
    
    def export_to_file(self, filename):
        np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)

    def gradient_check(self, X, Y, epsilon=1e-4):
        start_time = time.time()
        
        # gradient med backpropagation
        pred = self.predict(X)
        grads = self.update(X, Y, pred, rate=0) # uden at opdatere vægtene

        # gradient med finite difference
        params = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
        grad_approx = {}
        original_params = {}
        for param in params:
            theta = getattr(self, param)
            grad_approx[param] = np.zeros_like(theta)
            original_params[param] = np.copy(theta)
            it = np.nditer(theta, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                theta_plus = np.copy(theta)
                theta_minus = np.copy(theta)
                theta_plus[idx] += epsilon
                theta_minus[idx] -= epsilon

                setattr(self, param, theta_plus)
                J_plus = self.loss_function(X, Y)

                setattr(self, param, theta_minus)
                J_minus = self.loss_function(X, Y)

                grad_approx[param][idx] = (J_plus - J_minus) / (2 * epsilon)
                setattr(self, param, theta)  # sæt parameteren tilbage til original værdi før næste iteration
                it.iternext()
            setattr(self, param, original_params[param])  # sæt parameteren tilbage til original værdi

        # Compare gradients
        for param in params:
            grads_diff = np.abs(grads[param] - grad_approx[param])
            grads_sum = np.abs(grads[param]) + np.abs(grad_approx[param]) + 1e-8
            relative_difference = grads_diff / grads_sum 
            print(f"Gradient check for {param}: min = {np.min(relative_difference):.10f}, max = {np.max(relative_difference):.10f}, mean = {np.mean(relative_difference):.10f}")
            
            # Flatten the arrays for easier iteration
            flat_diff = relative_difference.flatten()
            flat_indices = np.ndindex(relative_difference.shape)
            
            #save the individual parameter differences in a txt file
            with open(f"{param}_diff.txt", "w") as f:
                for idx, diff in zip(flat_indices, flat_diff):
                    f.write(f"{diff:.10f}\n")
                            
            
        
        end_time = time.time()
        print(f"Gradient check execution time: {end_time - start_time:.4f} seconds")

    def loss_function(self, X, Y):
        pred = self.predict(X)
        A3 = pred['A3']
        cost = -np.mean(np.sum(Y * np.log(A3 + 1e-8), axis=1))
        return cost

nn = NeuralNetwork()
nn.gradient_check(x_train[:5], y_train_one_hot[:5])  # Test the gradient before training
nn.train(x_train, y_train_one_hot, epochs=10000)