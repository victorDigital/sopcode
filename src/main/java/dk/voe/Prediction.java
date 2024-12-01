package dk.voe;

public class Prediction {
  double[][] A1;
  double[][] A2;
  double[][] A3;
  double[][] Z1;
  double[][] Z2;
  double[][] Z3;

  public Prediction(double[][] A1, double[][] A2, double[][] A3, double[][] Z1, double[][] Z2, double[][] Z3) {
    this.A1 = A1;
    this.A2 = A2;
    this.A3 = A3;
    this.Z1 = Z1;
    this.Z2 = Z2;
    this.Z3 = Z3;
  }
}