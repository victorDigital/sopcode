package dk.voe;

import java.util.ArrayList;

import processing.core.PApplet;
import processing.core.PVector;

public class LayoutManager extends Drawable {
  ArrayList<Window> windows;
  ArrayList<PVector> windowPositions;
  ArrayList<PVector> windowSizes;

  public LayoutManager(PApplet p, PVector position) {
    super(p, position.x, position.y);
    windows = new ArrayList<Window>();
    windowPositions = new ArrayList<PVector>();
    windowSizes = new ArrayList<PVector>();
  }

  public void addWindow(Window window) {
    windows.add(window);
  }

  private void inferPositionsAndSizes() {
    windowPositions.clear();
    windowSizes.clear();
    // if one window, set it to fill the screen with a position of 0, 0
    // if two windows, set them to be side by side using a split view object
    // if three windows, set them to be in a 2x2 grid
    // if four windows, set them to be in a 2x2 grid

    // if more than 4 windows, remove the first window
    if (windows.size() > 4) {
      windows.remove(0);
    }

    if (windows.size() == 1) {
      windowPositions.add(new PVector(0, 0));
      windowSizes.add(new PVector(p.width, p.height));
    } else if (windows.size() == 2) {
      windowPositions.add(new PVector(0, 0));
      windowSizes.add(new PVector(p.width / 2, p.height));
      windowPositions.add(new PVector(p.width / 2, 0));
      windowSizes.add(new PVector(p.width / 2, p.height));
    } else if (windows.size() == 3) {
      windowPositions.add(new PVector(0, 0));
      windowSizes.add(new PVector(p.width / 2, p.height / 2));
      windowPositions.add(new PVector(p.width / 2, 0));
      windowSizes.add(new PVector(p.width / 2, p.height / 2));
      windowPositions.add(new PVector(0, p.height / 2));
      windowSizes.add(new PVector(p.width, p.height / 2));
    } else if (windows.size() == 4) {
      windowPositions.add(new PVector(0, 0));
      windowSizes.add(new PVector(p.width / 2, p.height / 2));
      windowPositions.add(new PVector(p.width / 2, 0));
      windowSizes.add(new PVector(p.width / 2, p.height / 2));
      windowPositions.add(new PVector(0, p.height / 2));
      windowSizes.add(new PVector(p.width / 2, p.height / 2));
      windowPositions.add(new PVector(p.width / 2, p.height / 2));
      windowSizes.add(new PVector(p.width / 2, p.height / 2));
    }
  }

  public void draw() {
    inferPositionsAndSizes();
    for (int i = 0; i < windows.size(); i++) {
      windows.get(i).setPosition(windowPositions.get(i));
      windows.get(i).setSize(windowSizes.get(i));
      windows.get(i).draw();
    }
  }

  public void update() {
    for (Window window : windows) {
      window.update();
    }
  }
}

class SplitView {
  Window window1;
  Window window2;
  PVector position;
  PVector size;

  boolean horizontal = true;

  SplitView(Window window1, Window window2, PVector position, PVector size, boolean horizontal) {
    this.window1 = window1;
    this.window2 = window2;
    this.position = position;
    this.size = size;
    this.horizontal = horizontal;

    window1.setPosition(position);
    if (horizontal) {
      window1.setSize(new PVector(size.x / 2, size.y));
      window2.setPosition(new PVector(position.x + size.x / 2, position.y));
      window2.setSize(new PVector(size.x / 2, size.y));
    } else {
      window1.setSize(new PVector(size.x, size.y / 2));
      window2.setPosition(new PVector(position.x, position.y + size.y / 2));
      window2.setSize(new PVector(size.x, size.y / 2));
    }
  }

}
