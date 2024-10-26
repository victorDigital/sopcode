package dk.voe;

import java.util.ArrayList;

import processing.core.PApplet;
import processing.core.PVector;

class Window extends Drawable {
  protected PVector size;
  private PVector targetPosition;
  private PVector targetSize;
  protected float smallestDimension;
  protected SharedData sharedData;

  public Window(PVector position, PVector size, PApplet p, SharedData sharedData) {
    super(p, position.x, position.y);
    this.size = size;
    this.sharedData = sharedData;
  }

  public void draw() {
    // to be overwritten
  }

  public void update() {
    // to be overwritten
  }

  public void setPosition(PVector position) {
    targetPosition = position;
    this.position.add(PVector.sub(targetPosition, this.position).div(10));
  }

  public void setSize(PVector size) {
    targetSize = size;
    this.size.add(PVector.sub(targetSize, this.size).div(10));
    smallestDimension = size.x < size.y ? size.x : size.y;
  }

  protected void drawHeader(String title) {
    p.fill(255);
    p.textSize(32);
    p.text(title, 10, 40);
    p.line(10, 50, size.x - 10, 50);
  }

  protected double[][][] convertToArray3d(ArrayList<ArrayList<ArrayList<Double>>> list) {
    double[][][] array = new double[list.size()][][];
    for (int i = 0; i < list.size(); i++) {
      array[i] = new double[list.get(i).size()][];
      for (int j = 0; j < list.get(i).size(); j++) {
        array[i][j] = new double[list.get(i).get(j).size()];
        for (int k = 0; k < list.get(i).get(j).size(); k++) {
          array[i][j][k] = list.get(i).get(j).get(k);
        }
      }
    }
    return array;
  }

  protected double[][] convertToArray2d(ArrayList<ArrayList<Double>> list) {
    double[][] array = new double[list.size()][];
    for (int i = 0; i < list.size(); i++) {
      array[i] = new double[list.get(i).size()];
      for (int j = 0; j < list.get(i).size(); j++) {
        array[i][j] = list.get(i).get(j);
      }
    }
    return array;
  }

  protected ArrayList<ArrayList<ArrayList<Double>>> convertFromArray3d(double[][][] array) {
    ArrayList<ArrayList<ArrayList<Double>>> list = new ArrayList<>();
    for (int i = 0; i < array.length; i++) {
      list.add(new ArrayList<>());
      for (int j = 0; j < array[i].length; j++) {
        list.get(i).add(new ArrayList<>());
        for (int k = 0; k < array[i][j].length; k++) {
          list.get(i).get(j).add(array[i][j][k]);
        }
      }
    }
    return list;
  }

  protected ArrayList<ArrayList<Double>> convertFromArray2d(double[][] array) {
    ArrayList<ArrayList<Double>> list = new ArrayList<>();
    for (int i = 0; i < array.length; i++) {
      list.add(new ArrayList<>());
      for (int j = 0; j < array[i].length; j++) {
        list.get(i).add(array[i][j]);
      }
    }
    return list;
  }
}