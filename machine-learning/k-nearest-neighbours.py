import numpy as np
from collections import Counter
from typing import List, Dict

class KNNClassifier:
    """
    k-Nearest Neighbors (k-NN) Classifier

    Attributes:
        k (int): Number of nearest neighbors to consider.
        training_data (Dict[str, List[np.ndarray]]): Dictionary where keys are class labels
            and values are lists of vectors corresponding to each class.
    """
    def __init__(self, training_data: Dict[str, List[np.ndarray]], k: int):
        if k % 2 == 0:
            raise ValueError("k should be odd.")
        self.k = k
        self.training_data = training_data

    def predict(self, test_vectors: np.ndarray) -> List[str]:
        """
        Predict the labels for the provided data.

        Args:
            test_vectors (np.ndarray): Data to classify.

        Returns:
            List[str]: Predicted labels.
        """
        return [self._predict_single(test_vector) for test_vector in test_vectors]

    def _predict_single(self, query_vector: np.ndarray) -> str:
        """
        Predict the label for a single data point.

        Args:
            query_vector (np.ndarray): Data point to classify.

        Returns:
            str: Predicted class label.
        """
        distances = []
        for class_label, training_vectors in self.training_data.items():
            for training_vector in training_vectors:
                distance = np.linalg.norm(training_vector - query_vector)
                distances.append((class_label, distance))
        
        distances.sort(key=lambda x: x[1])
        k_nearest = distances[:self.k]
        
        count = Counter([label for label, _ in k_nearest])
        most_common_label = count.most_common(1)[0][0]
        
        return most_common_label

# Example usage:
if __name__ == "__main__":
    # Sample training data (dictionary with class labels as keys and lists of vectors as values)
    training_data = {
        'A': [np.array([1, 2]), np.array([2, 3])],
        'B': [np.array([3, 4]), np.array([4, 5])]
    }

    knn = KNNClassifier(training_data, k=3)

    print("\nTraining data:")
    for label, vectors_list in training_data.items():
        for vector in vectors_list:
            print(f"{vector} -> class {label}")

    test_vectors = np.array([[1.5, 2.5], [3.5, 4.5]])
    predictions = knn.predict(test_vectors)

    print("\nPredictions:")
    for test_vector, prediction in zip(test_vectors, predictions):
        print(f"{test_vector} -> class {prediction}")
    print("\n")
