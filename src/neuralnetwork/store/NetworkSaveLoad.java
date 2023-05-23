package neuralnetwork.store;

import neuralnetwork.NeuralNetwork;

import java.io.*;

public class NetworkSaveLoad {
    public static DataOutputStream saveNetwork(NeuralNetwork neuralNetwork, File dest) throws IOException {
        DataOutputStream fos = new DataOutputStream(new FileOutputStream(dest));

        for(double weight : neuralNetwork.getWeights()) {
            fos.writeDouble(weight);
        }

        int[] layers = neuralNetwork.getLayers();

        fos.writeInt(layers.length);
        for(int layer : layers) {
            fos.writeInt(layer);
        }

        return fos;
    }

    public static NeuralNetwork loadNetwork(File src, NetworkConstructor networkConstructor) throws IOException {
        DataInputStream fis = new DataInputStream(new FileInputStream(src));

        int[] layers = new int[fis.readInt()];
        for(int layerIndex = 0; layerIndex != layers.length; layerIndex++) {
            layers[layerIndex] = fis.readInt();
        }

        return networkConstructor.construct(layers, fis);
    }
}
