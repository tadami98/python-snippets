import numpy as np
from collections import Counter
from typing import List, Union, Tuple

class KNNClassifier:
    """
    k-Nearest Neighbors (k-NN) Classifier

    Attributes:
        k (int): Number of nearest neighbors to consider.
        training_data (np.ndarray): Training data, with each row representing a data point.
        training_labels (np.ndarray): Labels corresponding to the training data.
    """
    def __init__(self, vectors: np.ndarray, labels: np.ndarray, k: int):
        if(k%2 == 0):
            raise ValueError("k should be odd.")
        self.k = k
        self.training_data = vectors
        self.training_labels = labels

    def predict(self, vectors: np.ndarray) -> List[str]:
        """
        Predict the labels for the provided data.

        Args:
            vectors (np.ndarray): Data to classify.

        Returns:
            List[str]: Predicted labels.
        """
        return [self._predict_single(vector) for vector in vectors]

    def _predict_single(self, vector: np.ndarray) -> str:
        """
        Predict the label for a single data point.

        Args:
            vector (np.ndarray): Data point to classify.

        Returns:
            str: Predicted class label.
        """
        distances = np.linalg.norm(self.training_data - vector, axis=1)
        k_nearest_indices = distances.argsort()[:self.k]
        k_nearest_labels = self.training_labels[k_nearest_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Example usage:
if __name__ == "__main__":
    # Sample training data (4 data points with 2 features each)
    vectors = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    labels = np.array(['A', 'A', 'B', 'B'])

    knn = KNNClassifier(vectors, labels, k=3)

    print("\nTraining data:")
    for vector, label in zip(vectors, labels):
        print(f"{vector} -> class {label}")

    vectors_test = np.array([[1.5, 2.5], [3.5, 4.5]])
    predictions = knn.predict(vectors_test)

    print("\nPredictions:")
    for vector, prediction in zip(vectors_test, predictions):
        if isinstance(prediction, tuple):
            print(f"{vector} -> classes {prediction} (tie)")
        else:
            print(f"{vector} -> class {prediction}")
    print("\n")
