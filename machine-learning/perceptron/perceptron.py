import numpy as np
from typing import Tuple

GREEN = '\033[92m'
RED = '\033[91m'
WHITE = '\033[0m'

class Perceptron:
    """
    Discrete unipolar perceptron -> y ∈ {0,1}
    
    Attributes:
        alpha (float): Learning rate.
        inputs (np.ndarray): Input values.
        weights (np.ndarray): Weights for inputs.
        threshold (float): Threshold value.
    """
    def __init__(self, alpha: float, inputs: np.ndarray, weights: np.ndarray, threshold: float):
        self.alpha = alpha
        self.inputs = inputs
        self.weights = weights
        self.threshold = threshold
        self.output = None

    def train(self, training_data: np.ndarray, targets: np.ndarray, epochs: int) -> None:
        """
        Train the perceptron on the given training data.
        
        Args:
            training_data (np.ndarray): Input training data.
            targets (np.ndarray): Target values for training.
            epochs (int): Number of training epochs.
        """
        for _ in range(epochs):
            for i in range(training_data.shape[0]):
                self.inputs = training_data[i]
                self._calculate_output()
                self._update_weights(targets[i])
                self._update_threshold(targets[i])

    def predict(self, inputs: np.ndarray) -> int:
        """
        Predict the output based on the given inputs.
        
        Args:
            inputs (np.ndarray): Input values for prediction.
        
        Returns:
            int: Predicted output.
        """
        self.inputs = inputs
        self._calculate_output()
        return self.output

    def calculate_metrics(self, test_data: np.ndarray, test_targets: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Calculate metrics for the model based on test data.
        
        Args:
            test_data (np.ndarray): Test input data.
            test_targets (np.ndarray): Target values for test data.
        
        Returns:
            Tuple[float, float, float, float]: Accuracy, precision, recall, and F-score.
        """
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
        """
        Calculate the output based on inputs, weights, and threshold.
        """
        net = np.dot(self.inputs, self.weights)
        self.output = 1 if net >= self.threshold else 0
        
    def _update_weights(self, target: int) -> None:
        """
        Update the weights based on the target output.
        
        Args:
            target (int): Target output value.
        """
        delta_weights = self.alpha * (target - self.output) * self.inputs
        self.weights += delta_weights

    def _update_threshold(self, target: int) -> None:
        """
        Update the threshold based on the target output.
        
        Args:
            target (int): Target output value.
        """
        delta_threshold = self.alpha * (target - self.output)
        self.threshold -= delta_threshold

    @staticmethod
    def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from a file and return inputs and targets.
        
        Args:
            filename (str): Path to the data file.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Input data and target values.
        """
        data = np.genfromtxt(filename, delimiter=',', dtype=str)
        inputs = data[:, :-1].astype(float)
        labels = data[:, -1]
        targets = np.array([1 if label == 'Iris-versicolor' else 0 for label in labels])
        return inputs, targets
    
def main():
    try:
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
            formatted_sample = np.array2string(sample, formatter={'float_kind': lambda x: f"{x:.1f}"})
            
            if predicted_label == actual_label:
                predicted_text = f"{GREEN}->  Predicted: {predicted_label}{WHITE}"
            else:
                predicted_text = f"{RED}->  Predicted: {predicted_label}{WHITE}"
    
            print(f"{formatted_sample} - {actual_label:<16} {predicted_text}")

        # Calculate and print metrics
        accuracy, precision, recall, fscore = perceptron.calculate_metrics(test_data, test_targets)
        print(f"""
        Metrics:
        Accuracy: {accuracy * 100:.2f}%
        Precision: {precision * 100:.2f}%
        Recall: {recall * 100:.2f}%
        F-score: {fscore * 100:.2f}%
        """)
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
