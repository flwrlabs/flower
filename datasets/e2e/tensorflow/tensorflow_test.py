import unittest

import numpy as np
import tensorflow as tf
from datasets.utils.logging import disable_progress_bar
from parameterized import parameterized_class, parameterized
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

from flwr_datasets import FederatedDataset


def SimpleCNN():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model


@parameterized_class(
    [
        {"dataset_name": "uoft-cs/cifar10", "test_split": "test"},
        {"dataset_name": "uoft-cs/cifar10", "test_split": "test"},
    ]
)
class FdsToTensorFlow(unittest.TestCase):
    """Test the conversion from FDS to PyTorch Dataset and Dataloader."""

    dataset_name = ""
    test_split = ""
    expected_img_shape_after_transform = [32, 32, 3]

    @classmethod
    def setUpClass(cls):
        """Disable progress bar to keep the log clean."""
        disable_progress_bar()

    def _create_tensorflow_dataset(self, batch_size: int) -> tf.data.Dataset:
        """Create a tensorflow dataset from the FederatedDataset."""
        partition_id = 0
        fds = FederatedDataset(dataset=self.dataset_name, partitioners={"train": 100})
        partition = fds.load_partition(partition_id, "train")
        tf_dataset = partition.to_tf_dataset(columns="img", label_cols="label",
                                             batch_size=batch_size,
                                             shuffle=False)
        return tf_dataset

    def test_create_partition_dataset_shape(self) -> None:
        """Test if the DataLoader returns batches with the expected shape."""
        batch_size = 16
        dataset = self._create_tensorflow_dataset(batch_size)
        batch = next(iter(dataset))
        images = batch[0]
        self.assertEqual(tuple(images.shape),
                         (batch_size, *self.expected_img_shape_after_transform))

    def test_create_partition_dataloader_with_transforms_batch_type(self) -> None:
        """Test if the DataLoader returns batches of type dictionary."""
        batch_size = 16
        dataset = self._create_tensorflow_dataset(batch_size)
        batch = next(iter(dataset))
        self.assertIsInstance(batch, tuple)

    def test_create_partition_dataloader_with_transforms_data_type(self) -> None:
        """Test to verify if the data in the DataLoader batches are of type Tensor."""
        batch_size = 16
        dataset = self._create_tensorflow_dataset(batch_size)
        batch = next(iter(dataset))
        images = batch[0]
        self.assertIsInstance(images, tf.Tensor)

    @parameterized.expand([
        ("not_nan", np.isnan),
        ("not_inf", np.isinf),
    ])
    def test_train_model_loss_value(self, name, condition_func):
        model = SimpleCNN()
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        dataset = self._create_tensorflow_dataset(16)

        # Perform a single epoch of training
        history = model.fit(dataset, epochs=1, verbose=0)

        # Fetch the last loss from history
        last_loss = history.history['loss'][-1]

        # Check if the last loss is NaN or Infinity
        self.assertFalse(condition_func(last_loss))


if __name__ == '__main__':
    unittest.main()
