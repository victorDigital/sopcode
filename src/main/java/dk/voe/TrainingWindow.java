package dk.voe;

import processing.core.PApplet;
import processing.core.PVector;

public class TrainingWindow extends Window {

  Button startButton;
  Button stopButton;

  NeuralNetwork nn;

  Graph costGraph;

  public TrainingWindow(PVector position, PVector size, PApplet p, SharedData sharedData) {
    super(position, size, p, sharedData);
    nn = new NeuralNetwork(new PVector(50, 50), p, new int[] { 28 * 28, 16, 16, 10 });
    costGraph = new Graph(p, new PVector(10, 100), new PVector(300, 200), position);
    nn.loadDataSetToMemory();

    long startTime = System.currentTimeMillis();
    PApplet.println("dataset cost:" + nn.computeLoss());
    long endTime = System.currentTimeMillis();
    PApplet.println("Time taken: " + (endTime - startTime) + " ms");

    startButton = new Button(p, 10f, 60f, 60, 40, "Start Training", position, () -> {
      nn.trainNeuralNetwork(100);
      return null;
    });
    stopButton = new Button(p, 260f, 60f, 60, 40, "Stop Training", position, () -> {
      return null;
    });

  }

  public void draw() {
    p.pushMatrix();
    p.translate(position.x, position.y);
    drawHeader("Training");
    startButton.display();
    stopButton.display();
    costGraph.draw();
    p.popMatrix();
  }

  public void update() {
    if (p.mousePressed) {
      startButton.setPositionAndOffset(new PVector(10, 60), position);
      stopButton.setPositionAndOffset(new PVector(260, 60), position);
      costGraph.setPositionAndOffset(new PVector(10, 100), position);
      startButton.onClick();
      stopButton.onClick();
    }
  }

  /*
   * public void sendModel() {
   * sharedData.weights = convertFromArray3d(nn.getWeights());
   * sharedData.biases = convertFromArray2d(nn.getBiases());
   * }
   */
}
