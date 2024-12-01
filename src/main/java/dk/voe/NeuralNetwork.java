package dk.voe;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import dk.voe.MnistDataLoader.DataSet;
import processing.core.PApplet;
import processing.core.PVector;

public class NeuralNetwork extends Drawable {
  protected DataSet[] dataset;
  static Data dat = new Data();
  static Parameters par = new Parameters();

  public NeuralNetwork(PVector position, PApplet p, int[] layerSizes) {
    super(p, position.x, position.y);
    initializeParameters();
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

  // Activation function (sigmoid)
  public static double phi(double x) {
    return 1.0 / (1.0 + Math.exp(-x));
  }

  // Derivative of sigmoid function
  public static double dphi(double x) {
    double fx = phi(x);
    return fx * (1 - fx);
  }

  // Softmax function
  public static double[] softmax(double[] x) {
    double max = Arrays.stream(x).max().getAsDouble();
    double[] expX = new double[x.length];
    double sum = 0.0;
    for (int i = 0; i < x.length; i++) {
      expX[i] = Math.exp(x[i] - max);
      sum += expX[i];
    }
    for (int i = 0; i < x.length; i++) {
      expX[i] /= sum;
    }
    return expX;
  }

  public double[] predict(double[] inputs) {
    dat.X = new double[1][inputs.length];
    dat.X[0] = inputs;
    Prediction pred = predict(par);
    return pred.A3[0];
  }

  // Predict function (forward propagation)
  private Prediction predict(Parameters par) {
    // Z1 = X * W1 + b1
    double[][] Z1 = addBias(matrixMultiply(dat.X, par.W1), par.b1);
    double[][] A1 = applyFunction(Z1, x -> phi(x));
    // Z2 = A1 * W2 + b2
    double[][] Z2 = addBias(matrixMultiply(A1, par.W2), par.b2);
    double[][] A2 = applyFunction(Z2, x -> phi(x));
    // Z3 = A2 * W3 + b3
    double[][] Z3 = addBias(matrixMultiply(A2, par.W3), par.b3);
    double[][] A3 = new double[Z3.length][Z3[0].length];
    for (int i = 0; i < Z3.length; i++) {
      A3[i] = softmax(Z3[i]);
    }
    return new Prediction(A1, A2, A3, Z1, Z2, Z3);
  }

  // Update function (backpropagation and parameter update)
  public void update(Prediction pred, Parameters par, double rate) {
    int m = dat.X.length;
    double[][] A1 = pred.A1;
    double[][] A2 = pred.A2;
    double[][] A3 = pred.A3;
    // dZ3 = A3 - Y
    double[][] dZ3 = subtractMatrices(A3, dat.Y);
    // dW3 = (A2' * dZ3) / m
    double[][] dW3 = scalarDivide(matrixMultiply(transposeMatrix(A2), dZ3), m);
    // db3 = sum of dZ3 columns / m
    double[] db3 = scalarDivide(sumColumns(dZ3), m);
    // dZ2 = (dZ3 * W3') * dphi(Z2)
    double[][] dZ2 = elementWiseMultiply(matrixMultiply(dZ3, transposeMatrix(par.W3)),
        applyFunction(pred.Z2, x -> dphi(x)));
    // dW2 = (A1' * dZ2) / m
    double[][] dW2 = scalarDivide(matrixMultiply(transposeMatrix(A1), dZ2), m);
    // db2 = sum of dZ2 columns / m
    double[] db2 = scalarDivide(sumColumns(dZ2), m);
    // dZ1 = (dZ2 * W2') * dphi(Z1)
    double[][] dZ1 = elementWiseMultiply(matrixMultiply(dZ2, transposeMatrix(par.W2)),
        applyFunction(pred.Z1, x -> dphi(x)));
    // dW1 = (X' * dZ1) / m
    double[][] dW1 = scalarDivide(matrixMultiply(transposeMatrix(dat.X), dZ1), m);
    // db1 = sum of dZ1 columns / m
    double[] db1 = scalarDivide(sumColumns(dZ1), m);
    // Update parameters
    par.W1 = subtractMatrices(par.W1, scalarMultiply(dW1, rate));
    par.b1 = subtractVectors(par.b1, scalarMultiply(db1, rate));
    par.W2 = subtractMatrices(par.W2, scalarMultiply(dW2, rate));
    par.b2 = subtractVectors(par.b2, scalarMultiply(db2, rate));
    par.W3 = subtractMatrices(par.W3, scalarMultiply(dW3, rate));
    par.b3 = subtractVectors(par.b3, scalarMultiply(db3, rate));
  }

  // Train neural network
  public void trainNeuralNetwork(int epochs) {
    for (int epoch = 1; epoch <= epochs; epoch++) {
      Prediction pred = predict(par);
      update(pred, par, 1.0);
      if (epoch % 10 == 0) {
        double loss = computeLoss();
        double accuracy = computeAccuracy();
        System.out.println("Epoch " + epoch + " Loss: " + loss + ", Accuracy: " + accuracy + "%");
      }
    }
  }

  // Initialize parameters
  public void initializeParameters() {
    Random rand = new Random();
    dat.nI = 784; // Number of input features
    dat.nO = 10; // Number of output classes
    int[] noNodes = dat.noNodes;

    par.W1 = randomMatrix(dat.nI, noNodes[0], rand);
    par.b1 = randomVector(noNodes[0], rand);
    par.W2 = randomMatrix(noNodes[0], noNodes[1], rand);
    par.b2 = randomVector(noNodes[1], rand);
    par.W3 = randomMatrix(noNodes[1], dat.nO, rand);
    par.b3 = randomVector(dat.nO, rand);
  }

  // Utility function to create a random matrix
  public double[][] randomMatrix(int rows, int cols, Random rand) {
    double[][] mat = new double[rows][cols];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        mat[i][j] = -1 + 2 * rand.nextDouble(); // Random values between -1 and 1
      }
    }
    return mat;
  }

  // Utility function to create a random vector
  public static double[] randomVector(int size, Random rand) {
    double[] vec = new double[size];
    for (int i = 0; i < size; i++) {
      vec[i] = -1 + 2 * rand.nextDouble(); // Random values between -1 and 1
    }
    return vec;
  }

  // Matrix multiplication
  public static double[][] matrixMultiply(double[][] A, double[][] B) {
    // A is of shape (m, p) and B is of shape (p, n)
    int m = A.length;
    int n = B[0].length;
    int p = A[0].length;
    double[][] result = new double[m][n];
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        double sum = 0.0;
        for (int k = 0; k < p; k++) {
          sum += A[i][k] * B[k][j];
        }
        result[i][j] = sum;
      }
    }
    return result;
  }

  // Transpose matrix
  public static double[][] transposeMatrix(double[][] A) {
    int m = A.length;
    int n = A[0].length;
    double[][] result = new double[n][m];
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        result[j][i] = A[i][j];
      }
    }
    return result;
  }

  // Add bias vector to matrix
  public static double[][] addBias(double[][] A, double[] b) {
    int m = A.length;
    int n = A[0].length;
    double[][] result = new double[m][n];
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        result[i][j] = A[i][j] + b[j];
      }
    }
    return result;
  }

  // Apply function element-wise to matrix
  public static double[][] applyFunction(double[][] A, java.util.function.Function<Double, Double> func) {
    int m = A.length;
    int n = A[0].length;
    double[][] result = new double[m][n];
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        result[i][j] = func.apply(A[i][j]);
      }
    }
    return result;
  }

  // Element-wise multiplication of matrices
  public static double[][] elementWiseMultiply(double[][] A, double[][] B) {
    int m = A.length;
    int n = A[0].length;
    double[][] result = new double[m][n];
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        result[i][j] = A[i][j] * B[i][j];
      }
    }
    return result;
  }

  // Subtract matrices
  public static double[][] subtractMatrices(double[][] A, double[][] B) {
    int m = A.length;
    int n = A[0].length;
    double[][] result = new double[m][n];
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        result[i][j] = A[i][j] - B[i][j];
      }
    }
    return result;
  }

  // Subtract vectors
  public static double[] subtractVectors(double[] A, double[] B) {
    int n = A.length;
    double[] result = new double[n];
    for (int i = 0; i < n; i++) {
      result[i] = A[i] - B[i];
    }
    return result;
  }

  // Scalar multiply matrix
  public static double[][] scalarMultiply(double[][] A, double scalar) {
    int m = A.length;
    int n = A[0].length;
    double[][] result = new double[m][n];
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        result[i][j] = A[i][j] * scalar;
      }
    }
    return result;
  }

  // Scalar multiply vector
  public static double[] scalarMultiply(double[] A, double scalar) {
    int n = A.length;
    double[] result = new double[n];
    for (int i = 0; i < n; i++) {
      result[i] = A[i] * scalar;
    }
    return result;
  }

  // Scalar divide matrix
  public static double[][] scalarDivide(double[][] A, double scalar) {
    return scalarMultiply(A, 1.0 / scalar);
  }

  // Scalar divide vector
  public static double[] scalarDivide(double[] A, double scalar) {
    return scalarMultiply(A, 1.0 / scalar);
  }

  // Sum columns of a matrix
  public static double[] sumColumns(double[][] A) {
    int m = A.length;
    int n = A[0].length;
    double[] sums = new double[n];
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int i = 0; i < m; i++) {
        sum += A[i][j];
      }
      sums[j] = sum;
    }
    return sums;
  }

  // loss function
  public double computeLoss() {
    List<double[]> images = dataset[0].getImages();
    List<double[]> labels = oneHot(dataset[0].getLabels());
    double loss = 0;
    for (int i = 0; i < images.size(); i++) {
      dat.X = new double[1][784];
      dat.Y = new double[1][10];
      dat.X[0] = images.get(i);
      dat.Y[0] = labels.get(i);
      Prediction pred = predict(par);
      double[][] A3 = pred.A3;
      loss += -Math.log(A3[0][argMax(dat.Y[0])]);
    }
    return loss / images.size();
  }

  // Compute accuracy
  public double computeAccuracy() {
    List<double[]> images = dataset[1].getImages();
    List<Integer> labels = dataset[1].getLabels();
    int correct = 0;
    for (int i = 0; i < images.size(); i++) {
      dat.X = new double[1][784];
      dat.Y = new double[1][10];
      dat.X[0] = images.get(i);
      dat.Y[0] = oneHot(labels.get(i));
      Prediction pred = predict(par);
      double[][] A3 = pred.A3;
      System.out.println(argMax(A3[0]));
      if (argMax(A3[0]) == labels.get(i)) {
        correct++;
      }
    }
    return 100.0 * (correct / images.size());
  }

  // Find index of maximum value in array
  public static int argMax(double[] array) {
    int index = 0;
    double max = array[0];
    for (int i = 1; i < array.length; i++) {
      if (array[i] > max) {
        max = array[i];
        index = i;
      }
    }
    return index;
  }

  public static double[] oneHot(int label) {
    double[] oneHot = new double[10];
    oneHot[label] = 1;
    return oneHot;
  }

  public static List<double[]> oneHot(List<Integer> labels) { // runs oneHot for each label in the list and returns a
                                                              // list of oneHot labels
    return labels.stream().map(label -> oneHot(label)).collect(Collectors.toList());
  }

  public void setPosition(PVector position) {
    this.position = position;
  }

  public double hashDat(Data dat) {
    // a way of seeing if the dat have changed
    return Arrays.hashCode(dat.X[0]) + Arrays.hashCode(dat.Y[0]);
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

}