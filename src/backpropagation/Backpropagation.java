package backpropagation;

@FunctionalInterface
public interface Backpropagation {
    double[] backpropagation(double[] errors, double[] neuronResults);
}
