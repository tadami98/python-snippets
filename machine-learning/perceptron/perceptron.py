import numpy as np

class Perceptron:
    """Discrete unipolar perceptron -> y ∈ {0,1}"""
    def __init__(self, alpha: float, inputs: np.ndarray, weights: np.ndarray, threshold: float):
        self.alpha = alpha
        self.inputs = inputs
        self.weights = weights
        self.threshold = threshold
        self.output = None

    def train(self, training_data: np.ndarray, targets: np.ndarray, epochs: int) -> None:
        for _ in range(epochs):
            for i in range(training_data.shape[0]):
                self.inputs = training_data[i]
                self._calculate_output()
                self._modify_weights(targets[i])
                self._modify_threshold(targets[i])

    def predict(self, inputs: np.ndarray) -> int:
        self.inputs = inputs
        self._calculate_output()
        return self.output

    def calculate_metrics(self, test_data: np.ndarray, test_targets: np.ndarray):
        predictions = [self.predict(sample) for sample in test_data]
        true_positives = sum((pred == 1) and (true == 1) for pred, true in zip(predictions, test_targets))
        true_negatives = sum((pred == 0) and (true == 0) for pred, true in zip(predictions, test_targets))
        false_positives = sum((pred == 1) and (true == 0) for pred, true in zip(predictions, test_targets))
        false_negatives = sum((pred == 0) and (true == 1) for pred, true in zip(predictions, test_targets))
        
        accuracy = (true_positives + true_negatives) / len(test_targets)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
        fscore = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        
        return accuracy, precision, recall, fscore
    
    def _calculate_output(self) -> None:
        net = np.dot(self.inputs, self.weights)
        if net >= self.threshold:
            self.output = 1
        else:
            self.output = 0
        
    def _modify_weights(self, target: int) -> None:
        delta_weights = self.alpha * (target - self.output) * self.inputs
        self.weights += delta_weights

    def _modify_threshold(self, target: int) -> None:
        delta_threshold = self.alpha * (target - self.output)
        self.threshold -= delta_threshold

    @staticmethod
    def load_data(filename: str):
        data = np.genfromtxt(filename, delimiter=',', dtype=str)
        inputs = data[:, :-1].astype(float)
        labels = data[:, -1]
        targets = np.array([1 if label == 'Iris-versicolor' else 0 for label in labels])
        return inputs, targets
    
class main():
    # Load training data
    training_data, training_targets = Perceptron.load_data('training.csv')
    
    # Initialize weights and threshold
    initial_weights = np.zeros(training_data.shape[1])
    initial_threshold = 0.5
    learning_rate = 0.1

    # Initialize and train the perceptron
    perceptron = Perceptron(alpha=learning_rate, 
                            inputs=np.zeros(training_data.shape[1]), 
                            weights=initial_weights, 
                            threshold=initial_threshold)
    
    perceptron.train(training_data, training_targets, epochs=10)
    
    test_data, test_targets = Perceptron.load_data('test.csv')

    for sample, target in zip(test_data, test_targets):
        output = perceptron.predict(sample)
        predicted_label = 'Iris-versicolor' if output == 1 else 'Iris-setosa'
        actual_label = 'Iris-versicolor' if target == 1 else 'Iris-setosa'
        formatted_sample = np.array2string(sample, formatter={'float_kind':lambda x: f"{x:.1f}"})
        print(f"Input: {formatted_sample}\nPredicted: {predicted_label},\nActual:    {actual_label}\n")

    # Calculate and print metrics
    accuracy, precision, recall, fscore = perceptron.calculate_metrics(test_data, test_targets)
    print(f"\nMetrics:\nAccuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF-score: {fscore:.2f}")

if __name__ == '__main__':
    main()