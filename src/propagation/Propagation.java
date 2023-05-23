package propagation;

import neuralnetwork.NeuralNetwork;

public interface Propagation {
    double[] propagation(double[] inputs, NeuralNetwork network);
}
