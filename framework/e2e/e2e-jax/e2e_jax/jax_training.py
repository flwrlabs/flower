"""Linear Regression with JAX.

This code examples is based on the following code example:
https://coax.readthedocs.io/en/latest/examples/linear_regression/jax.html

If you have any questions concerning the linear regression used with jax
please read the JAX documentation or the mentioned tutorial.
"""

from typing import Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

key = jax.random.PRNGKey(0)


def load_data() -> (
    Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]
):
    # Load dataset
    X, y = make_regression(n_features=3, random_state=0)
    X, X_test, y, y_test = train_test_split(X, y)
    return X, y, X_test, y_test


def load_model(model_shape) -> Dict:
    # Extract model parameters
    params = {"b": jax.random.uniform(key), "w": jax.random.uniform(key, model_shape)}
    return params


def loss_fn(params, X, y) -> Callable:
    # Return MSE as loss
    err = jnp.dot(X, params["w"]) + params["b"] - y
    return jnp.mean(jnp.square(err))


def train(params, grad_fn, X, y) -> Tuple[np.array, float, int]:
    num_examples = X.shape[0]
    for epochs in range(50):
        grads = grad_fn(params, X, y)
        params = jax.tree_util.tree_map(lambda p, g: p - 0.05 * g, params, grads)
        loss = loss_fn(params, X, y)
        if epochs % 10 == 0:
            print(f"For Epoch {epochs} loss {loss}")
    return params, loss, num_examples


def evaluation(params, grad_fn, X_test, y_test) -> Tuple[float, int]:
    num_examples = X_test.shape[0]
    err_test = loss_fn(params, X_test, y_test)
    loss_test = jnp.mean(jnp.square(err_test))
    return loss_test, num_examples


def main():
    X, y, X_test, y_test = load_data()
    model_shape = X.shape[1:]
    grad_fn = jax.grad(loss_fn)
    print("Model Shape", model_shape)
    params = load_model(model_shape)
    print("Params", params)
    params, loss, num_examples = train(params, grad_fn, X, y)
    print("Training loss:", loss)
    loss, num_examples = evaluation(params, grad_fn, X_test, y_test)
    print("Evaluation loss:", loss)


if __name__ == "__main__":
    main()
