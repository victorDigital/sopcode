package dk.voe;

import java.util.ArrayList;

public class SharedData {
  public ArrayList<ArrayList<ArrayList<Double>>> weights;
  public ArrayList<ArrayList<Double>> biases;

  public SharedData() {
    weights = new ArrayList<ArrayList<ArrayList<Double>>>();
    biases = new ArrayList<ArrayList<Double>>();
  }
}