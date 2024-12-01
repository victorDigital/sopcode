import pygame
import sys
import random
import math
import subprocess

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Drawing Pad")

BUTTON_COLOR = (70, 70, 70)
WHITE = (255, 255, 255) 
BLACK = (0, 0, 0)

import struct
from array import array
from os.path import join


# MNIST Data Loader Class
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath: str, labels_filepath: str) -> tuple[list[list[list[int]]], list[int]]:
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([[0] * cols for _ in range(rows)])
        for i in range(size):
            for r in range(rows):
                for c in range(cols):
                    images[i][c][r] = image_data[i * rows * cols + r * cols + c]
        
        return images, labels
            
    def load_data(self) -> tuple[tuple[list[list[list[int]]], list[int]], tuple[list[list[list[int]]], list[int]]]:
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)

class DrawingPad:
    def __init__(self):
        self.dx = 28
        self.dy = 28
        self.last_pixel_x = None
        self.last_pixel_y = None
        self.pixel_size = min(WIDTH // self.dx, HEIGHT // self.dy)
        self.pixel_values = [
            [0 for _ in range(self.dy)] for _ in range(self.dx)]

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            grid_x = x // self.pixel_size
            grid_y = y // self.pixel_size
            if (self.last_pixel_x == grid_x and self.last_pixel_y == grid_y):
                return
            if 0 <= grid_x < self.dx and 0 <= grid_y < self.dy:
                self.last_pixel_x = grid_x
                self.last_pixel_y = grid_y
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        new_x = grid_x + i
                        new_y = grid_y + j
                        if 0 <= new_x < self.dx and 0 <= new_y < self.dy:
                            distance = (
                                (new_x - grid_x) ** 2 + (new_y - grid_y) ** 2
                            ) ** 0.5
                            newPixelValue = min(
                                255,
                                self.pixel_values[new_x][new_y]
                                + int(255 * (1 - distance / 3)),
                            )
                            if newPixelValue > self.pixel_values[new_x][new_y]:
                                self.pixel_values[new_x][new_y] = newPixelValue

    def draw(self):
        for i in range(self.dx):
            for j in range(self.dy):
                color_value = self.pixel_values[i][j]
                rect = pygame.Rect(
                    i * self.pixel_size,
                    j * self.pixel_size,
                    self.pixel_size,
                    self.pixel_size,
                )
                pygame.draw.rect(
                    screen, (color_value, color_value, color_value), rect)
                pygame.draw.rect(screen, (50, 50, 50), rect, 1)
    
    def clear(self):
        self.pixel_values = [
            [0 for _ in range(self.dy)] for _ in range(self.dx)
        ]
    
    def set_image(self, image):
        self.pixel_values = image
    
    def get_image(self):
        return self.pixel_values

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def dsigmoid(x: float) -> float:
    return math.exp(-x) / (1 + math.exp(-x))**2

def softmax(x: list[float]) -> list[float]:
    exp_x = [math.exp(i - max(x)) for i in x]
    sum_exp_x = sum(exp_x)
    return [i / sum_exp_x for i in exp_x]

def matrix_multiply(m1: list[list[float]], m2: list[list[float]]) -> list[list[float]]:
    rows, cols, common = len(m1), len(m2[0]), len(m2)
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            for k in range(common):
                out[i][j] += m1[i][k] * m2[k][j]
    return out

def matrix_subtract(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    rows, cols = len(A), len(A[0])
    result = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            result[i][j] = A[i][j] - B[i][j]
    return result

def matrix_add(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    rows, cols = len(A), len(A[0])
    result = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            result[i][j] = A[i][j] + B[i][j]
    return result

def matrix_transpose(A: list[list[float]]) -> list[list[float]]:
    rows, cols = len(A), len(A[0])
    result = [[0.0] * rows for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            result[j][i] = A[i][j]
    return result

def matrix_elementwise_multiply(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    rows, cols = len(A), len(A[0])
    result = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            result[i][j] = A[i][j] * B[i][j]
    return result

def vector_scalar_divide(vector: list[float], scalar: float) -> list[float]:
    return [v / scalar for v in vector]

def vector_scalar_multiply(vector: list[float], scalar: float) -> list[float]:
    return [v * scalar for v in vector]

def vector_subtract(A: list[float], B: list[float]) -> list[float]:
    return [a - b for a, b in zip(A, B)]

def matrix_sum_cols(A: list[list[float]]) -> list[float]:
    return [sum(col) for col in zip(*A)]

def matrix_scalar_multiply(A: list[list[float]], scalar: float) -> list[list[float]]:
    rows, cols = len(A), len(A[0])
    result = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            result[i][j] = A[i][j] * scalar
    return result

def matrix_scalar_divide(A: list[list[float]], scalar: float) -> list[list[float]]:
    rows, cols = len(A), len(A[0])
    result = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            result[i][j] = A[i][j] / scalar
    return result

def matrix_apply_function(A: list[list[float]], func: callable) -> list[list[float]]:
    rows, cols = len(A), len(A[0])
    result = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            result[i][j] = func(A[i][j])
    return result

def format_expected_output(y: int) -> list[float]:
    return [1.0 if i == y else 0.0 for i in range(10)]

def dphiofphi(phi: float) -> float:
    return phi * (1 - phi)

class NeuralNet:
    def __init__(self, layer_sizes):
        self.noNodes = layer_sizes[1:-1]
        self.nI = layer_sizes[0]
        self.nO = layer_sizes[-1]
        self.W1 = [[random.uniform(-1, 1) for _ in range(self.noNodes[0])] for _ in range(self.nI)]
        self.b1 = [random.uniform(-1, 1) for _ in range(self.noNodes[0])]
        self.W2 = [[random.uniform(-1, 1) for _ in range(self.noNodes[1])] for _ in range(self.noNodes[0])]
        self.b2 = [random.uniform(-1, 1) for _ in range(self.noNodes[1])]
        self.W3 = [[random.uniform(-1, 1) for _ in range(self.nO)] for _ in range(self.noNodes[1])]
        self.b3 = [random.uniform(-1, 1) for _ in range(self.nO)]

    def predict(self, X):
        Z1 = matrix_add(matrix_multiply(X, self.W1), [self.b1])
        A1 = matrix_apply_function(Z1, sigmoid)
        Z2 = matrix_add(matrix_multiply(A1, self.W2), [self.b2])
        A2 = matrix_apply_function(Z2, sigmoid)
        Z3 = matrix_add(matrix_multiply(A2, self.W3), [self.b3])
        A3 = [softmax(z) for z in Z3]
        return {'A1': A1, 'A2': A2, 'A3': A3, 'Z1': Z1, 'Z2': Z2, 'Z3': Z3}

    def update(self, pred, X, Y, rate=0.01):
        m = len(X)
        A1, A2, A3 = pred['A1'], pred['A2'], pred['A3']
        dZ3 = matrix_subtract(A3, Y)
        dW3 = matrix_scalar_divide(matrix_multiply(matrix_transpose(A2), dZ3), m)
        db3 = vector_scalar_divide(matrix_sum_cols(dZ3), m)
        dZ2_part = matrix_multiply(dZ3, matrix_transpose(self.W3))
        dZ2 = matrix_elementwise_multiply(dZ2_part, matrix_apply_function(A2, dphiofphi))
        dW2 = matrix_scalar_divide(matrix_multiply(matrix_transpose(A1), dZ2), m)
        db2 = vector_scalar_divide(matrix_sum_cols(dZ2), m)
        dZ1_part = matrix_multiply(dZ2, matrix_transpose(self.W2))
        dZ1 = matrix_elementwise_multiply(dZ1_part, matrix_apply_function(A1, dphiofphi))
        dW1 = matrix_scalar_divide(matrix_multiply(matrix_transpose(X), dZ1), m)
        db1 = vector_scalar_divide(matrix_sum_cols(dZ1), m)
        self.W1 = matrix_subtract(self.W1, matrix_scalar_multiply(dW1, rate))
        self.b1 = vector_subtract(self.b1, vector_scalar_multiply(db1, rate))
        self.W2 = matrix_subtract(self.W2, matrix_scalar_multiply(dW2, rate))
        self.b2 = vector_subtract(self.b2, vector_scalar_multiply(db2, rate))
        self.W3 = matrix_subtract(self.W3, matrix_scalar_multiply(dW3, rate))
        self.b3 = vector_subtract(self.b3, vector_scalar_multiply(db3, rate))

    def train_neural_network(self, X, Y, epochs):
        for epoch in range(1, epochs + 1):
            pred = self.predict(X)
            self.update(pred, X, Y, rate=1)
            if epoch % 10 == 0:
                loss = -sum(
                    sum(Y[i][j] * math.log(pred['A3'][i][j]) for j in range(len(Y[0])))
                    for i in range(len(Y))
                ) / len(Y)
                correct_predictions = sum(
                    1 for i in range(len(Y))
                    if Y[i].index(1) == pred['A3'][i].index(max(pred['A3'][i]))
                )
                cp = 100 * correct_predictions / len(Y)
                print(f"Epoch {epoch}, Loss: {loss}, Correct: {cp}%")

def main():
    #load the data
    mnist_dataloader = MnistDataloader(
        'mnist/train-images.idx3-ubyte', 'mnist/train-labels.idx1-ubyte',
        'mnist/t10k-images.idx3-ubyte', 'mnist/t10k-labels.idx1-ubyte'
    )
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    
    screen.fill(BLACK)
    drawing_pad = DrawingPad()
    clear_button = pygame.Rect(10, HEIGHT - 60, 100, 50)
    font = pygame.font.Font(None, 36)
    button_text = font.render("Clear", True, WHITE)
    
    
    # set the drawing pad to the first image
    drawing_pad.set_image(x_train[48773])
    
    # create the neural network
    nn = NeuralNet([28*28, 16, 16, 10])
    
    # train the neural network
    X = [[pixel for row in image for pixel in row] for image in x_train]
    Y = [format_expected_output(label) for label in y_train]
    nn.train_neural_network(X, Y, epochs=100)
        


    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and clear_button.collidepoint(
                event.pos
            ):
                drawing_pad.clear()

        mouse_clicked = pygame.mouse.get_pressed()[0]
        if mouse_clicked:
            drawing_pad.handle_event(
                pygame.event.Event(
                    pygame.MOUSEBUTTONDOWN, {"pos": pygame.mouse.get_pos()}
                )
            )
        screen.fill(BLACK)
        drawing_pad.draw()
        pygame.draw.rect(screen, BUTTON_COLOR, clear_button)
        screen.blit(button_text, (clear_button.x + 10, clear_button.y + 10))
        pygame.display.flip()


if __name__ == "__main__":
    main()