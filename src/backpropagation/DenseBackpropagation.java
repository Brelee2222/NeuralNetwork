package backpropagation;

import neuralnetwork.NeuralNetwork;

public class DenseBackpropagation implements Backpropagation {
    public final NeuralNetwork neuralNetwork;
    public double learningRate;

    public double getLearningRate() {
        return learningRate;
    }

    public DenseBackpropagation(NeuralNetwork neuralNetwork, double learningRate) {
        this.learningRate = learningRate;
        this.neuralNetwork = neuralNetwork;
    }

    @Override
    public double[] backpropagation(double[] errors, double[] neuronResults) {
        double[] weights = neuralNetwork.getWeights();
        int[] layers = neuralNetwork.getLayers();

        double learningRate = getLearningRate();

        int neuronResultsIndex = neuronResults.length - errors.length;
        int weightsIndex = weights.length;

        for(int layerIndex = layers.length - 2; layerIndex != -1; layerIndex--) {

            for(int errorIndex = errors.length-1; errorIndex != -1; errorIndex--) {
                double result = neuronResults[neuronResultsIndex + errorIndex];

                errors[errorIndex] *= result * (1 - result);
            }

            double[] nextErrors = new double[layers[layerIndex]];

            neuronResultsIndex -= nextErrors.length;

            for(int errorIndex = errors.length-1; errorIndex != -1; errorIndex--) {
                double errorSignal = errors[errorIndex];

                weightsIndex -= nextErrors.length;

                for(int nextErrorsIndex = nextErrors.length-1; nextErrorsIndex != -1; nextErrorsIndex--) {
                    nextErrors[nextErrorsIndex] += weights[weightsIndex + nextErrorsIndex] * errorSignal;

                    weights[weightsIndex + nextErrorsIndex] += learningRate * neuronResults[neuronResultsIndex + nextErrorsIndex] * errorSignal;
                }

                weights[--weightsIndex] += learningRate * errorSignal;
            }

            errors = nextErrors;
        }

        return errors;
    }
}
