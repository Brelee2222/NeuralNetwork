package propagation;

import neuralnetwork.NeuralNetwork;

public class SigmoidPropagation implements Propagation {

    public double activate(double input) {
        return 1/(1 + Math.exp(-input));
    }

    @Override
    public double[] propagation(double[] inputs, NeuralNetwork network) {
        int[] layers = network.getLayers();
        double[] weights = network.getWeights();

        int weightsIndex = 1;

        double[] nextInputs;


        for(int layerIndex = 1; layerIndex != layers.length; layerIndex++) {

            nextInputs = new double[layers[layerIndex]];

            for(int nextInputsIndex = 0; nextInputsIndex != nextInputs.length; nextInputsIndex++) {
                double input = weights[weightsIndex-1];

                for(int inputIndex = 0; inputIndex != inputs.length; inputIndex++) {
                    input += inputs[inputIndex] * weights[weightsIndex + inputIndex];
                }

                weightsIndex += inputs.length + 1;

                nextInputs[nextInputsIndex] = activate(input);
            }

            inputs = nextInputs;
        }

        return inputs;
    }
}
