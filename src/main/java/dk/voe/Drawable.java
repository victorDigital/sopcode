package dk.voe;

import processing.core.PApplet;
import processing.core.PVector;

public class Drawable {
  protected PApplet p;
  protected PVector position = new PVector(0, 0);

  public Drawable(PApplet p, float x, float y) {
    position.set(x, y);
    this.p = p;
  }
}
