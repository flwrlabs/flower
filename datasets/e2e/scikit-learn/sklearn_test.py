import unittest

import numpy as np
from parameterized import parameterized_class
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from flwr_datasets import FederatedDataset


# Using parameterized testing, two different sets of preprocessing:
# 1. Without scaling.
# 2. With standard scaling.
@parameterized_class(
    [
        {"dataset_name": "mnist", "preprocessing": None},
        {"dataset_name": "mnist", "preprocessing": StandardScaler()},
    ]
)
class FdsWithSKLearn(unittest.TestCase):
    """Test Flower Datasets with Scikit-learn's Logistic Regression."""

    dataset_name = ""
    preprocessing = None

    def _get_partition_data(self):
        """Retrieve partition data."""
        partition_id = 0
        fds = FederatedDataset(dataset=self.dataset_name, partitioners={"train": 10})
        partition = fds.load_partition(partition_id, "train")
        partition.set_format("numpy")
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        X_train, y_train = partition_train_test["train"][:]["image"], partition_train_test[
            "train"][:]["label"]
        X_test, y_test = partition_train_test["test"][:]["image"], partition_train_test[
            "test"][:]["label"]
        X_train = X_train.reshape(-1, 28 * 28)
        X_test = X_test.reshape(-1, 28 * 28)
        if self.preprocessing:
            self.preprocessing.fit(X_train)
            X_train = self.preprocessing.transform(X_train)
            X_test = self.preprocessing.transform(X_test)

        return X_train, X_test, y_train, y_test

    def test_data_shape(self):
        """Test if the data shape is maintained after preprocessing."""
        X_train, _, _, _ = self._get_partition_data()
        self.assertEqual(X_train.shape, (4_800, 28 * 28))

    def test_X_train_type(self):
        """Test if the data type is correct."""
        X_train, _, _, _ = self._get_partition_data()
        self.assertIsInstance(X_train, np.ndarray)

    def test_y_train_type(self):
        """Test if the data type is correct."""
        _, _, y_train, _ = self._get_partition_data()
        self.assertIsInstance(y_train, np.ndarray)

    def test_X_test_type(self):
        """Test if the data type is correct."""
        _, X_test, _, _ = self._get_partition_data()
        self.assertIsInstance(X_test, np.ndarray)

    def test_y_test_type(self):
        """Test if the data type is correct."""
        _, _, _, y_test = self._get_partition_data()
        self.assertIsInstance(y_test, np.ndarray)

    def test_train_classifier(self):
        """Test if the classifier trains without errors."""
        X_train, X_test, y_train, y_test = self._get_partition_data()
        try:
            clf = LogisticRegression()
            clf.fit(X_train, y_train)
        except Exception as e:
            self.fail(f"Fitting Logistic Regression raised {type(e)} unexpectedly!")

    def test_predict_from_classifier(self):
        """Test if the classifier predicts without errors."""
        X_train, X_test, y_train, y_test = self._get_partition_data()
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        try:
            _ = clf.predict(X_test)
        except Exception as e:
            self.fail(
                f"Predicting using Logistic Regression model raised {type(e)} "
                f"unexpectedly!")


if __name__ == '__main__':
    unittest.main()
