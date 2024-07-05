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

    @classmethod
    def from_user_input(cls):
        """
        Prompt user for class labels and vectors to create training data.

        Returns:
            KNNClassifier: Instance of KNNClassifier with user-defined training data.
        """
        training_data = {}
        k = int(input("Value of k: "))
        while k % 2 == 0:
            print("k should be odd. Please enter an odd number.")
            k = int(input("Value of k: "))

        while True:
            class_label = input("Class label: ").strip()
            if class_label == "":
                break
            
            vectors = cls._parse_vectors_input(input(f"Vectors: ").strip())
            if vectors is None:
                continue
            
            training_data[class_label] = vectors
        
        return cls(training_data, k)

    @staticmethod
    def _parse_vectors_input(vector_input: str) -> List[np.ndarray]:
        """
        Parse vector input in the format '[x1, y1] [x2, y2] ...'

        Args:
            vector_input (str): Input string containing vectors.

        Returns:
            List[np.ndarray]: List of parsed numpy arrays representing vectors.
        """
        vectors = []
        try:
            # Split vector input by ']' '[', then handle spaces
            vector_strings = [v.strip(' []') for v in vector_input.split('] [')]
            
            for v_str in vector_strings:
                # Split by comma and convert to float
                coords = v_str.split(',')
                vector = np.array([float(coord.strip()) for coord in coords])
                if len(vector) != 2:
                    raise ValueError
                vectors.append(vector)
        except (ValueError, IndexError):
            print("Invalid input format. Please enter vectors in the format '[x1, y1] [x2, y2] ...'")
            return None
        
        return vectors

# A -> [1, 2] [2, 3]       B -> [3, 4] [4, 5]       ? -> [1.5, 2.5] [3.5, 4.5]
if __name__ == "__main__":
    knn = KNNClassifier.from_user_input()

    print("\nTraining data:")
    for label, vectors_list in knn.training_data.items():
        for vector in vectors_list:
            print(f"{vector} -> class {label}")

    while True:
        test_vectors_input = input("\nVectors to evaluate: ").strip()
        if test_vectors_input == "":
            break
        
        test_vectors = KNNClassifier._parse_vectors_input(test_vectors_input)
        
        predictions = knn.predict(np.array(test_vectors))

        print("\nPredictions:")
        for test_vector, prediction in zip(test_vectors, predictions):
            print(f"{test_vector} -> class {prediction}")
