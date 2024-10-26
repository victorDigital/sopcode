package dk.voe;

import java.util.ArrayList;

import processing.core.PApplet;
import processing.core.PVector;

public class Graph extends Drawable {
  PVector size;
  PVector offset = new PVector(0, 0);

  PVector startLimit = new PVector(0, 0);
  PVector endLimit = new PVector(10, 10);

  ArrayList<GraphPoint> points = new ArrayList<GraphPoint>();

  public Graph(PApplet p, PVector position, PVector size, PVector offset) {
    super(p, position.x, position.y);
    this.size = size;
    this.offset = offset;
  }

  public void draw() {
    p.fill(50);
    p.stroke(150);
    p.rect(position.x, position.y, size.x, size.y);
    scaleGraph();
    drawGraph();
  }

  public void setPositionAndOffset(PVector pos, PVector offset) {
    this.position = pos;
    this.offset = offset;
  }

  private void scaleGraph() {
    float xMin = Float.MAX_VALUE;
    float xMax = Float.MIN_VALUE;
    float yMin = Float.MAX_VALUE;
    float yMax = Float.MIN_VALUE;

    for (GraphPoint point : points) {
      xMin = Math.min(xMin, point.position.x - 1);
      xMax = Math.max(xMax, point.position.x + 1);
      yMin = Math.min(yMin, point.position.y - 1);
      yMax = Math.max(yMax, point.position.y + 1);
    }

    p.println(xMin, xMax, yMin, yMax);

    startLimit = new PVector(xMin, yMin);
    endLimit = new PVector(xMax, yMax);
  }

  private void drawGraph() {
    p.stroke(255);
    p.noFill();
    p.beginShape();
    for (GraphPoint point : points) {
      float x = PApplet.map(point.position.x, startLimit.x, endLimit.x, position.x, position.x + size.x);
      float y = PApplet.map(point.position.y, startLimit.y, endLimit.y, position.y + size.y, position.y);
      p.vertex(x, y);
    }
    p.endShape();
  }

  public void addPoint(GraphPoint point) {
    points.add(point);
  }
}

final class GraphPoint {
  public PVector position;

  GraphPoint(PVector position) {
    this.position = position;
  }
}