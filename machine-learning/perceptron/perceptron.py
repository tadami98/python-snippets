import numpy as np
from typing import Tuple, Callable

GREEN = '\033[92m'
ORANGE = '\033[93m'
RED = '\033[91m'
WHITE = '\033[0m'

# Activation functions
def step_function(x: float) -> float:
    return 1 if x >= 0 else 0

def linear_function(x: float) -> float:
    return x

def sigmoid_function(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def tanh_function(x: float) -> float:
    return np.tanh(x)

class Perceptron:
    """
    Discrete unipolar perceptron -> y ∈ {0,1}
    
    Attributes:
        alpha (float): Learning rate.
        weights (np.ndarray): Weights for inputs.
        threshold (float): Threshold value.
        activation_func (Callable[[float], float]): Activation function for output calculation.
    """
    def __init__(self, learning_rate: float, num_inputs: int, activation_func: Callable[[float], float]):
        self.alpha = learning_rate
        self.weights = np.random.randn(num_inputs)  # Initialize weights randomly
        self.threshold = 0.5  # Initialize threshold (can be adjusted)
        self.activation_func = activation_func

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
        Calculate the output based on inputs, weights, threshold, and activation function.
        """
        net = np.dot(self.inputs, self.weights)
        activated_net = self.activation_func(net)
        self.output = 1 if activated_net >= self.threshold else 0
        
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
        
        # Initialize and train the perceptron with different activation functions
        learning_rate = 0.1
        num_inputs = training_data.shape[1]

        activation_functions = {
            'step': step_function,
            'linear': linear_function,
            'sigmoid': sigmoid_function,
            'tanh': tanh_function
        }

        print("")
        for name, activation_func in activation_functions.items():
            perceptron = Perceptron(learning_rate=learning_rate, num_inputs=num_inputs, activation_func=activation_func)
            perceptron.train(training_data, training_targets, epochs=10)
            
            # Load test data
            test_data, test_targets = Perceptron.load_data('test.csv')

            # Calculate metrics
            accuracy, precision, recall, fscore = perceptron.calculate_metrics(test_data, test_targets)
            print(f"{name.capitalize()} activation function:")
            print(f"Accuracy: {colorize_percentage(accuracy)}")
            print(f"Precision: {colorize_percentage(precision)}")
            print(f"Recall: {colorize_percentage(recall)}")
            print(f"F-score: {colorize_percentage(fscore)}\n")

    except Exception as e:
        print(f"An error occurred: {e}")

def colorize_percentage(value: float) -> str:
    """
    Colorize the percentage value based on predefined thresholds.

    Args:
        value (float): The percentage value to be colorized.

    Returns:
        str: The colorized percentage string.
    """
    if value == 1.0: return f"{GREEN}{value * 100:.2f}%{WHITE}"  # Green for 100.00%
    elif value >= 0.5: return f"{ORANGE}{value * 100:.2f}%{WHITE}"  # Orange for >= 50%
    else: return f"{RED}{value * 100:.2f}%{WHITE}"  # Red for < 50%

if __name__ == '__main__':
    main()
