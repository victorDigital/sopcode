package dk.voe;

//oversat til java fra python: https://www.kaggle.com/code/hojjatk/read-mnist-dataset

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

public class MnistDataLoader {
  private String trainingImagesFilepath;
  private String trainingLabelsFilepath;
  private String testImagesFilepath;
  private String testLabelsFilepath;

  public MnistDataLoader(String trainingImagesFilepath, String trainingLabelsFilepath,
      String testImagesFilepath, String testLabelsFilepath) {
    this.trainingImagesFilepath = trainingImagesFilepath;
    this.trainingLabelsFilepath = trainingLabelsFilepath;
    this.testImagesFilepath = testImagesFilepath;
    this.testLabelsFilepath = testLabelsFilepath;
  }

  private DataSet readImagesLabels(String imagesFilepath, String labelsFilepath) throws IOException {
    // Read labels
    List<Integer> labels = new ArrayList<>();
    try (FileInputStream labelsFile = new FileInputStream(labelsFilepath)) {
      byte[] magicBytes = new byte[4];
      byte[] sizeBytes = new byte[4];

      labelsFile.read(magicBytes);
      labelsFile.read(sizeBytes);

      int magic = ByteBuffer.wrap(magicBytes).order(ByteOrder.BIG_ENDIAN).getInt();
      int size = ByteBuffer.wrap(sizeBytes).order(ByteOrder.BIG_ENDIAN).getInt();

      if (magic != 2049) {
        throw new IOException("Magic number mismatch in labels file, expected 2049, got " + magic);
      }

      byte[] labelData = new byte[size];
      labelsFile.read(labelData);
      for (byte label : labelData) {
        labels.add((int) label & 0xFF);
      }
    }

    // Read images
    List<double[]> images = new ArrayList<>();
    try (FileInputStream imagesFile = new FileInputStream(imagesFilepath)) {
      byte[] magicBytes = new byte[4];
      byte[] sizeBytes = new byte[4];
      byte[] rowBytes = new byte[4];
      byte[] colBytes = new byte[4];

      imagesFile.read(magicBytes);
      imagesFile.read(sizeBytes);
      imagesFile.read(rowBytes);
      imagesFile.read(colBytes);

      int magic = ByteBuffer.wrap(magicBytes).order(ByteOrder.BIG_ENDIAN).getInt();
      int size = ByteBuffer.wrap(sizeBytes).order(ByteOrder.BIG_ENDIAN).getInt();
      int rows = ByteBuffer.wrap(rowBytes).order(ByteOrder.BIG_ENDIAN).getInt();
      int cols = ByteBuffer.wrap(colBytes).order(ByteOrder.BIG_ENDIAN).getInt();

      if (magic != 2051) {
        throw new IOException("Magic number mismatch in images file, expected 2051, got " + magic);
      }

      for (int i = 0; i < size; i++) {
        int[][] image = new int[rows][cols];
        byte[] imageData = new byte[rows * cols];
        imagesFile.read(imageData);

        for (int r = 0; r < rows; r++) {
          for (int c = 0; c < cols; c++) {
            image[r][c] = imageData[r * cols + c] & 0xFF;
          }
        }
        double[] flatImage = new double[rows * cols];
        for (int r = 0; r < rows; r++) {
          for (int c = 0; c < cols; c++) {
            flatImage[r * cols + c] = image[r][c];
          }
        }
        images.add(flatImage);
      }
    }

    return new DataSet(images, labels);
  }

  public DataSet[] loadData() throws IOException {
    DataSet training = readImagesLabels(trainingImagesFilepath, trainingLabelsFilepath);
    DataSet test = readImagesLabels(testImagesFilepath, testLabelsFilepath);
    return new DataSet[] { training, test };
  }

  public static class DataSet {
    private final List<double[]> images;
    private final List<Integer> labels;

    public DataSet(List<double[]> images, List<Integer> labels) {
      this.images = images;
      this.labels = labels;
    }

    public List<double[]> getImages() {
      return images;
    }

    public List<Integer> getLabels() {
      return labels;
    }
  }
}