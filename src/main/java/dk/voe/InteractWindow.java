package dk.voe;

import processing.core.PApplet;
import processing.core.PVector;

public class InteractWindow extends Window {
  DrawingPad drawingPad;
  NeuralNetwork nn;

  public InteractWindow(PVector position, PVector size, PApplet p, SharedData sharedData) {
    super(position, size, p, sharedData);
    drawingPad = new DrawingPad(new PVector(50, 50), new PVector(smallestDimension / 1.4f, smallestDimension / 1.4f),
        position, p);
    nn = new NeuralNetwork(new PVector(smallestDimension / 1.4f + 50, 70), p, new int[] { 28 * 28, 16, 16, 10 });

  }

  public void draw() {
    p.pushMatrix();
    p.translate(position.x, position.y);
    drawingPad.setSize(new PVector(smallestDimension / 1.4f, smallestDimension / 1.4f)); // force square

    drawHeader("Interact Window");

    drawingPad.draw();

    double[] inputs = drawingPad.getPixelsFlatDouble();
    double[] outputs = nn.predict(inputs);
    nn.renderPrediction(outputs);

    p.popMatrix();
  }

  public void update() {
    drawingPad.setOffset(position);
    nn.setPosition(new PVector(smallestDimension / 1.4f + 50, 70));
    drawingPad.update();

    // if the weights and biases in the shared data are the correct size, set the
    // model
    // if (sharedData.weights.size() > 0 && sharedData.biases.size() > 0) {
    // /* setModel(); */
    // sharedData.weights.clear();
    // sharedData.biases.clear();
    // }
  }

  /*
   * public void setModel() {
   * nn.setWeights(convertToArray3d(sharedData.weights));
   * nn.setBiases(convertToArray2d(sharedData.biases));
   * }
   */
}
