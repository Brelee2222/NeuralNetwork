package neuralnetwork.store;

import neuralnetwork.NeuralNetwork;

import java.io.DataInputStream;

@FunctionalInterface
public interface NetworkConstructor {
    NeuralNetwork construct(int[] layers, DataInputStream networkData);
}
