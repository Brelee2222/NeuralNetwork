package neuralnetwork;

import java.util.function.DoubleSupplier;
import java.util.stream.DoubleStream;

public class DenseNetwork implements NeuralNetwork {

    int[] layers;
    double[] weights;

    public DenseNetwork(DoubleSupplier weights, int... layers) {
        int totalWeights = layers[0], prevLayer = totalWeights;
        for(int layerIndex = 1; layerIndex != layers.length; layerIndex++) {
            totalWeights += prevLayer * (prevLayer = layers[layerIndex]);
        }

        this.weights = DoubleStream.generate(weights).limit(totalWeights).toArray();
        this.layers = layers.clone();
    }

    @Override
    public double[] getWeights() {
        return weights;
    }

    @Override
    public int[] getLayers() {
        return layers;
    }
}
