package dk.voe;

import java.io.IOException;
import java.util.List;

import dk.voe.MnistDataLoader.DataSet;
import processing.core.PApplet;
import processing.core.PVector;

public class NeuralNetwork extends Drawable {
  public Layer[] layers;
  protected DataSet[] dataset;

  public NeuralNetwork(PVector position, PApplet p, int[] layerSizes) {
    super(p, position.x, position.y);

    layers = new Layer[layerSizes.length - 1];
    for (int i = 0; i < layers.length; i++) {
      layers[i] = new Layer(layerSizes[i], layerSizes[i + 1], i);
    }

  }

  public double[] calculateOutputs(double[] inputs) {
    double[] outputs = inputs;
    for (Layer layer : layers) {
      outputs = layer.calculateOutputs(outputs);
    }

    return outputs;
  }

  public void learn(double learningRate) {
    List<double[]> images = dataset[0].getImages();

    for (int i = 0; i < images.size(); i++) {
      updateAllGradients(i);
    }

    for (Layer layer : layers) {
      layer.applyGradients(learningRate);
    }
  }

  public int classify(double[] inputs) {
    double[] outputs = calculateOutputs(inputs);
    int maxIndex = 0;
    for (int i = 1; i < outputs.length; i++) {
      if (outputs[i] > outputs[maxIndex]) {
        maxIndex = i;
      }
    }
    return maxIndex;
  }

  private double singleCost(double[] inputs, double[] expectedOutputs) {

    double[] outputs = calculateOutputs(inputs);
    double cost = 0;
    for (int i = 0; i < outputs.length; i++) {
      cost += layers[layers.length - 1].nodeCost(outputs[i], expectedOutputs[i]);
    }

    return cost;
  }

  public double costOfImage(int imageIndex) {
    return singleCost(dataset[0].getImages().get(imageIndex), oneHot(dataset[0].getLabels().get(imageIndex)));
  }

  public double totalCost() {
    List<double[]> images = dataset[0].getImages();
    double cost = 0;
    for (int i = 0; i < images.size(); i++) {
      cost += costOfImage(i);
    }
    return cost;
  }

  public void renderPrediction(double[] outputs) {
    p.fill(255);
    p.textSize(20);

    // find the index of the output node with the highest value
    int maxIndex = 0;
    for (int i = 1; i < outputs.length; i++) {
      if (outputs[i] > outputs[maxIndex]) {
        maxIndex = i;
      }
    }

    for (int i = 0; i < outputs.length; i++) {
      if (i == maxIndex) {
        p.fill(0, 255, 0);
      } else {
        p.fill(255);
      }
      p.text(String.format("%d: %.1f%%", (i + 1) % 10, outputs[i] * 100), position.x, position.y + 20 * i);
    }
  }

  private double[] oneHot(int label) {
    double[] oneHot = new double[10];
    oneHot[label] = 1;
    return oneHot;
  }

  public void loadDataSetToMemory() {
    PApplet.println("Loading dataset to memory...");
    String basePath = "src/main/resources/mnist/";
    MnistDataLoader loader = new MnistDataLoader(
        basePath + "train-images.idx3-ubyte",
        basePath + "train-labels.idx1-ubyte",
        basePath + "t10k-images.idx3-ubyte",
        basePath + "t10k-labels.idx1-ubyte");

    try {
      dataset = loader.loadData();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public void setPosition(PVector position) {
    this.position = position;
  }

  public double[][][] getWeights() {
    double[][][] weights = new double[layers.length][][];
    for (int i = 0; i < layers.length; i++) {
      weights[i] = layers[i].weights;
    }
    return weights;
  }

  public double[][] getBiases() {
    double[][] biases = new double[layers.length][];
    for (int i = 0; i < layers.length; i++) {
      biases[i] = layers[i].biases;
    }
    return biases;
  }

  public void setWeights(double[][][] weights) {
    for (int i = 0; i < layers.length; i++) {
      layers[i].weights = weights[i];
    }
  }

  public void setBiases(double[][] biases) {
    for (int i = 0; i < layers.length; i++) {
      layers[i].biases = biases[i];
    }
  }

  public double calculateAccuracy() {
    int correct = 0;
    for (int i = 0; i < dataset[1].getImages().size(); i++) {
      double[] inputs = dataset[1].getImages().get(i);
      int expectedOutput = dataset[1].getLabels().get(i);
      int output = classify(inputs);
      if (output == expectedOutput) {
        correct++;
      }
    }
    return (double) correct / dataset[1].getImages().size();
  }

  public void updateAllGradients(int imageIndex) {
    double[] inputs = dataset[0].getImages().get(imageIndex);
    double[] expectedOutputs = oneHot(dataset[0].getLabels().get(imageIndex));

    calculateOutputs(inputs);

    Layer outputLayer = layers[layers.length - 1];
    double[] nodeValues = outputLayer.calculateOutputLayerNodeValues(expectedOutputs);
    outputLayer.updateGradients(nodeValues);

    for (int i = layers.length - 2; i >= 0; i--) {
      Layer oldLayer = layers[i + 1];
      double[] oldNodeValues = oldLayer.calculateHiddenLayerNodeValues(layers[i], nodeValues);
      layers[i].updateGradients(oldNodeValues);
      nodeValues = oldNodeValues;
    }

  }

}

class Layer {
  protected int numNodesIn, numNodesOut;
  public double[][] weights;
  public double[] biases;
  public double[][] weightCostGradients;
  public double[] biasCostGradients;
  public double[] nodeValues;

  public Layer(int numNodesIn, int numNodesOut, int layerIndex) {
    this.numNodesIn = numNodesIn;
    this.numNodesOut = numNodesOut;
    weights = new double[numNodesOut][numNodesIn];
    biases = new double[numNodesOut];
    weightCostGradients = new double[numNodesOut][numNodesIn];
    biasCostGradients = new double[numNodesOut];

    nodeValues = new double[numNodesOut];

    // randomize weights and biases in the layer
    for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
      biases[nodeOut] = Math.random() * 2 - 1;
      for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
        weights[nodeOut][nodeIn] = Math.random() * 2 - 1;
      }
    }
  }

  public void applyGradients(double learningRate) {
    for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
      biases[nodeOut] -= learningRate * biasCostGradients[nodeOut];
      for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
        weights[nodeOut][nodeIn] -= learningRate * weightCostGradients[nodeOut][nodeIn];
      }
    }
  }

  public double[] calculateOutputs(double[] inputs) {

    if (inputs.length != numNodesIn) {
      throw new IllegalArgumentException("Expected " + numNodesIn + " inputs, but got " + inputs.length);
    }
    double[] weightedInputs = new double[numNodesOut];

    for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
      double weightedInput = biases[nodeOut];
      for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
        weightedInput += weights[nodeOut][nodeIn] * inputs[nodeIn];
      }
      weightedInputs[nodeOut] = activationFunction(weightedInput);
    }

    return weightedInputs;
  }

  public double nodeCost(double outputActivation, double targetActivation) {
    double error = targetActivation - outputActivation;
    return error * error; // Quadratic cost
  }

  public double nodeCostDerivative(double outputActivation, double targetActivation) {
    return 2 * (outputActivation - targetActivation); // Quadratic cost
  }

  public double activationFunction(double x) {
    return 1 / (1 + Math.exp(-x)); // Sigmoid
  }

  public double activationDerivative(double x) {
    double a = activationFunction(x);
    return a * (1 - a);
  }

  public double[] calculateOutputLayerNodeValues(double[] expectedOutputs) {
    for (int i = 0; i < nodeValues.length; i++) {
      double costDerivative = nodeCostDerivative(expectedOutputs[i], nodeValues[i]);
      double activationDerivative = activationDerivative(nodeValues[i]);
      nodeValues[i] = costDerivative * activationDerivative;
    }

    return nodeValues;
  }

  public double[] calculateHiddenLayerNodeValues(Layer oldLayer, double[] oldNodeValues) {
    double[] newNodeValues = new double[numNodesOut];

    for (int newNodeIndex = 0; newNodeIndex < newNodeValues.length; newNodeIndex++) {
      double newNodeValue = 0;
      for (int oldNodeIndex = 0; oldNodeIndex < oldNodeValues.length; oldNodeIndex++) {
        double weightedInputDerivative = oldLayer.weights[oldNodeIndex][newNodeIndex];
        newNodeValue += oldNodeValues[oldNodeIndex] * weightedInputDerivative;
      }
      newNodeValue *= activationDerivative(newNodeValue);
      newNodeValues[newNodeIndex] = newNodeValue;
    }

    return newNodeValues;
  }

  public void updateGradients(double[] nodeValues) {
    for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
      if (nodeOut >= nodeValues.length)
        break; // Ensure nodeOut does not exceed nodeValues length
      for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
        double costWeightDerivative = nodeValues[nodeOut] * weights[nodeOut][nodeIn];
        weightCostGradients[nodeOut][nodeIn] = costWeightDerivative;
      }
      biasCostGradients[nodeOut] = nodeValues[nodeOut];
    }
  }

  public int getNumNodesIn() {
    return numNodesIn;
  }

  public int getNumNodesOut() {
    return numNodesOut;
  }

  public int getAmountOfBiases() {
    return biases.length;
  }
}