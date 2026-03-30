"""jaxexample: A Flower / JAX app."""

import warnings
from typing import Any, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax
from datasets.utils.logging import disable_progress_bar
from flax import linen as nn
from flax.training.train_state import TrainState
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from jax import Array

disable_progress_bar()

rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=6, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=16, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=120)(x)
        x = nn.relu(x)
        x = nn.Dense(features=84)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


def create_model(rng: Array) -> Tuple[CNN, Any]:
    cnn = CNN()
    return cnn, cnn.init(rng, jnp.ones([1, 28, 28, 1]))["params"]


def create_train_state(learning_rate: float) -> TrainState:
    """Creates initial `TrainState`."""

    tx = optax.sgd(learning_rate, momentum=0.9)
    model, model_params = create_model(rng)
    return TrainState.create(apply_fn=model.apply, params=model_params, tx=tx)


def get_params(params: Any) -> List[npt.NDArray[Any]]:
    """Get model parameters as list of numpy arrays."""
    return [np.array(param) for param in jax.tree_util.tree_leaves(params)]


def set_params(
    train_state: TrainState, global_params: Sequence[npt.NDArray[Any]]
) -> TrainState:
    """Create a new trainstate with the global_params."""
    new_params_dict = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(train_state.params), list(global_params)
    )
    return train_state.replace(params=new_params_dict)


@jax.jit
def apply_model(
    state: TrainState, images: Array, labels: Array
) -> Tuple[Any, Array, Array]:
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def update_model(state: TrainState, grads: Any) -> TrainState:
    return state.apply_gradients(grads=grads)


def train(state: TrainState, train_ds) -> Tuple[TrainState, float, float]:
    """Train for a single epoch."""

    epoch_loss = []
    epoch_accuracy = []

    for batch in train_ds:
        batch_images = batch["image"]
        batch_labels = batch["label"]
        grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, float(train_loss), float(train_accuracy)


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions, batch_size):
    """Load partition MNIST data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2)

    partition["train"].set_format("jax")
    partition["test"].set_format("jax")

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [
            jnp.expand_dims(jnp.float32(img), 3) / 255 for img in batch["image"]
        ]
        batch["label"] = [jnp.int16(label) for label in batch["label"]]
        return batch

    train_partition = (
        partition["train"]
        .batch(batch_size, num_proc=2, drop_last_batch=True)
        .with_transform(apply_transforms)
    )
    test_partition = (
        partition["test"]
        .batch(batch_size, num_proc=2, drop_last_batch=True)
        .with_transform(apply_transforms)
    )

    train_partition.shuffle(seed=1234)
    test_partition.shuffle(seed=1234)

    return train_partition, test_partition
