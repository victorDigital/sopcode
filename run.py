from array import array
import pygame
import sys
import numpy as np
from mnist_dataloader import MnistDataloader  # Added import statement
from activations import phi, softmax

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Drawing Pad")

BUTTON_COLOR = (70, 70, 70)
WHITE = (255, 255, 255) 
BLACK = (0, 0, 0)


# Load MNIST data
print("Loading MNIST data...")
mnist_dataloader = MnistDataloader(
        'mnist/train-images.idx3-ubyte', 'mnist/train-labels.idx1-ubyte',
        'mnist/t10k-images.idx3-ubyte', 'mnist/t10k-labels.idx1-ubyte'
    )
(x_train, y_train), _ = mnist_dataloader.load_data()
print("MNIST data loaded.")

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
        self.pixel_values = np.array(image).T
    
    def get_image(self):
        pixel_values = self.pixel_values
        image = np.array(pixel_values).T
        return image

def predict(image, W1, b1, W2, b2, W3, b3):
    x = np.array(image).reshape(-1) / 255.0
    Z1 = x @ W1 + b1
    A1 = phi(Z1)
    Z2 = A1 @ W2 + b2
    A2 = phi(Z2)
    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)
    return np.argmax(A3), A3


def main():
    # error if the model.npz file is not found in the same directory
    model = None
    W1, b1, W2, b2, W3, b3 = None, None, None, None, None, None
    try:
        model = np.load("model.npz")
        W1, b1 = model['W1'], model['b1']
        W2, b2 = model['W2'], model['b2']
        W3, b3 = model['W3'], model['b3']
    except FileNotFoundError:
        print("Model file not found. Please run train.py to train the model.")
        sys.exit(1)
    
    screen.fill(BLACK)
    drawing_pad = DrawingPad()
    clear_button = pygame.Rect(10, HEIGHT - 60, 100, 50)
    font = pygame.font.Font(None, 36)
    button_text = font.render("Clear", True, WHITE)
    predicted_digit = None
    probabilities = None
    
    # load img 1337 from the training set and set the image on the drawing pad
    drawing_pad.set_image(x_train[1337])
    ## print the label of the image
    print("Label of the image:", y_train[1337])

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and clear_button.collidepoint(
                event.pos
            ):
                drawing_pad.clear()

        image = drawing_pad.get_image()
        predicted_digit, probabilities = predict(image, W1, b1, W2, b2, W3, b3)
        print("Predicted digit:", predicted_digit)

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
        if predicted_digit is not None:
            prediction_text = font.render(f"Predicted: {predicted_digit}", True, WHITE)
            screen.blit(prediction_text, (WIDTH - 200, 10))
            for i, prob in enumerate(probabilities):
                prob_text = font.render(f"{i}: {prob * 100:.2f}%", True, WHITE)
                screen.blit(prob_text, (WIDTH - 200, 40 + i * 30))
            
        pygame.display.flip()


if __name__ == "__main__":
    main()