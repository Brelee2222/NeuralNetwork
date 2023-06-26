package propagation;

import neuralnetwork.NeuralNetwork;

public class LogisticPropagation implements Propagation {

    public final NeuralNetwork neuralNetwork;

    public SigmoidPropagation(NeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
    }

    @Override
    public double[] propagation(double[] inputs) {
        int[] layers = neuralNetwork.getLayers();
        double[] weights = neuralNetwork.getWeights();

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

                nextInputs[nextInputsIndex] = 1/(1 + Math.exp(-input));
            }

            inputs = nextInputs;
        }

        return inputs;
    }
}
