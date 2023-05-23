package backpropagation;

import neuralnetwork.NeuralNetwork;

public class DenseBackpropagation implements Backpropagation {
    public double learningRate;

    public double getLearningRate() {
        return learningRate;
    }

    @Override
    public double[] backpropagation(double[] errors, double[] neuronResults, NeuralNetwork neuralNetwork) {
        double[] weights = neuralNetwork.getWeights();
        int[] layers = neuralNetwork.getLayers();

        double learningRate = getLearningRate();

        int neuronResultsIndex = neuronResults.length - errors.length;
        int weightsIndex = weights.length;

        neuronResultsIndex -= errors.length;

        for(int layerIndex = layers.length - 2; layerIndex != -1; layerIndex--) {

            for(int errorIndex = errors.length-1; errorIndex != -1; errorIndex--) {
                double result = neuronResults[neuronResultsIndex + errorIndex];

                errors[errorIndex] = errors[errorIndex] * result * (1 - result);
            }

            double[] nextErrors = new double[layers[layerIndex]];

            neuronResultsIndex -= nextErrors.length;

            for(int errorIndex = errors.length-1; errorIndex != -1; errorIndex--) {
                double errorSignal = errors[errorIndex];

                weightsIndex -= nextErrors.length;

                for(int nextErrorsIndex = nextErrors.length-1; nextErrorsIndex != -1; nextErrorsIndex--) {
                    nextErrors[nextErrorsIndex] += errorSignal * weights[weightsIndex + nextErrorsIndex];

                    weights[weightsIndex + nextErrorsIndex] += errorSignal * neuronResults[neuronResultsIndex + nextErrorsIndex] * learningRate;
                }

                weights[--weightsIndex] += errorSignal * learningRate;
            }

            errors = nextErrors;
        }

        return errors;
    }
}
