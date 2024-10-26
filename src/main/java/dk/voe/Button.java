package dk.voe;

import java.util.concurrent.Callable;

import processing.core.PApplet;
import processing.core.PVector;

public class Button extends Drawable {
  private int w;
  private int h;
  private String text;
  private boolean clicked = false;
  private Callable<Void> action;
  private PVector offset;

  public Button(PApplet p, float x, float y, int w, int h, String text, PVector offset, Callable<Void> action) {
    super(p, x, y);
    this.text = text;
    this.action = action;
    this.w = w;
    this.h = h;
    this.offset = offset;
  }

  void display() {
    if (clicked) {
      p.fill(50, 50, 50);
    } else {
      p.fill(255);
    }

    p.rect(position.x, position.y, w + p.textWidth(text), h);
    p.fill(0);
    p.textSize(20);
    p.text(text, position.x + w / 2, position.y + h / 2);
  }

  void onClick() {
    float WindowLocalMouseX = p.mouseX - offset.x;
    float WindowLocalMouse = p.mouseY - offset.y;

    if (WindowLocalMouseX > position.x && WindowLocalMouseX < position.x + w + p.textWidth(text)
        && WindowLocalMouse > position.y
        && WindowLocalMouse < position.y + h) {
      clicked = true;
      try {
        p.println("Button clicked");
        action.call();
      } catch (Exception e) {
        e.printStackTrace();
      }
    } else {
      clicked = false;
    }
  }

  public void setPositionAndOffset(PVector pos, PVector offset) {
    this.position = pos;
    this.offset = offset;
  }
}