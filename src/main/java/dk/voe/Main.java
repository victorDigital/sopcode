package dk.voe;

import processing.core.PApplet;
import processing.core.PVector;

public class Main extends PApplet {
    LayoutManager layoutManager;
    SharedData sharedData;

    NeuralNetwork nn;

    public static void main(String[] args) {
        PApplet.main("dk.voe.Main", args);
    }

    public void settings() {
        size(1200, 600, P2D);
        sharedData = new SharedData(); // Example size
        layoutManager = new LayoutManager(this, new PVector(0, 0));
        layoutManager
                .addWindow(new InteractWindow(new PVector(0, 0), new PVector(width / 2, height), this, sharedData));
        layoutManager.addWindow(
                new TrainingWindow(new PVector(width / 2, 0), new PVector(width / 2, height), this, sharedData));
    }

    public void setup() {
        frameRate(144);
    }

    public void draw() {
        background(0);
        layoutManager.draw();
        layoutManager.update();
    }

    public void keyPressed() {
        if (key == 's') {
            // Handle key press
        }
    }
}