package backpropagation;

import neuralnetwork.NeuralNetwork;

import java.util.stream.DoubleStream;

@FunctionalInterface
public interface Backpropagation {
    double[] backpropagation(double[] errors, double[] neuronResults, NeuralNetwork neuralNetwork);
}
