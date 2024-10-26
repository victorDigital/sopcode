package dk.voe;

import processing.core.PApplet;
import processing.core.PVector;

public class DrawingPad extends Drawable {
  private int dx = 28, dy = 28;
  PVector size;
  PVector offset = new PVector(0, 0);
  private float[][] pixels = new float[dx][dy];

  private PVector pixelSize;
  private Button clearButton;

  DrawingPad(PVector position, PVector size, PVector offset, PApplet p) {
    super(p, position.x, position.y);
    this.size = size;
    this.offset = offset;
    this.pixelSize = new PVector(size.x / dx, size.y / dy);
    clearButton = new Button(p, position.x, position.y + size.y + 10, 100, 50, "Clear", offset, () -> {
      for (int i = 0; i < dx; i++) {
        for (int j = 0; j < dy; j++) {
          pixels[i][j] = 0;
        }
      }
      return null;
    });
  }

  void draw() {
    float ActualMouseX = p.mouseX - offset.x;
    float ActualMouseY = p.mouseY - offset.y;
    p.fill(255);
    p.stroke(100);
    // Draw all pixels

    for (int i = 0; i < dx; i++) {
      for (int j = 0; j < dy; j++) {
        boolean pixelHovered = PVector.dist(new PVector(ActualMouseX, ActualMouseY),
            new PVector(
                position.x + i * pixelSize.x + pixelSize.x / 2,
                position.y + j * pixelSize.y + pixelSize.y / 2)) < 20;
        float currentPixelColor = PApplet.map(pixels[i][j], 0, 1, 0, 255);
        p.fill(pixelHovered ? PApplet.constrain(currentPixelColor + 50, 0, 255) : currentPixelColor);
        p.rect(position.x + i * pixelSize.x, position.y + j * pixelSize.y, pixelSize.x, pixelSize.y);
      }
    }

    clearButton.display();

  }

  void update() {
    float ActualMouseX = p.mouseX - offset.x;
    float ActualMouseY = p.mouseY - offset.y;
    // if the mouse is pressed set the pixels to a mapped value based on the
    // distance from the mouse in a 20 pixel radius
    if (p.mousePressed) {
      clearButton.onClick();
      for (int i = 0; i < dx; i++) {
        for (int j = 0; j < dy; j++) {
          boolean isCloseEnough = PVector.dist(new PVector(ActualMouseX, ActualMouseY),
              new PVector(
                  position.x + i * pixelSize.x + pixelSize.x / 2,
                  position.y + j * pixelSize.y + pixelSize.y / 2)) < 20;
          if (isCloseEnough) {
            float newPixelValue = PApplet.map(PVector.dist(new PVector(ActualMouseX, ActualMouseY),
                new PVector(
                    position.x + i * pixelSize.x + pixelSize.x / 2,
                    position.y + j * pixelSize.y + pixelSize.y / 2)),
                0, 20, 1, 0);
            if (newPixelValue > pixels[i][j]) {
              pixels[i][j] = newPixelValue;
            }
          }
        }
      }
    }
  }

  public float[][] getPixels() {
    return pixels;
  }

  public float[] getPixelsFlat() {
    float[] flatPixels = new float[dx * dy];
    for (int i = 0; i < dx; i++) {
      for (int j = 0; j < dy; j++) {
        flatPixels[i * dx + j] = pixels[i][j];
      }
    }
    return flatPixels;
  }

  public double[] getPixelsFlatDouble() {
    float[] flatPixels = getPixelsFlat();
    double[] doublePixels = new double[flatPixels.length];
    for (int i = 0; i < flatPixels.length; i++) {
      doublePixels[i] = flatPixels[i];
    }
    return doublePixels;
  }

  public void setPixels(int[][] pixels) {
    for (int i = 0; i < dx; i++) {
      for (int j = 0; j < dy; j++) {
        this.pixels[i][j] = (float) pixels[i][j] / 255.0f;
      }
    }
  }

  public void setSize(PVector size) {
    this.size = size;
    this.pixelSize = new PVector(size.x / dx, size.y / dy);
    setOffset(offset);
  }

  public void setOffset(PVector offset) {
    this.offset = offset;
    clearButton.setPositionAndOffset(new PVector(position.x, position.y + size.y),
        offset);
  }
}