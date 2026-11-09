"""Define utility functions for TraceFL provenance analysis.

They are not directly relevant to the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""

import numpy as np
from sklearn import metrics


def compute_importance(n_elements, decay_factor=0.9):
    """Compute importance weights for a sequence of elements using a decay factor.

    Parameters
    ----------
    n_elements : int
        The number of elements for which importance is computed.
    decay_factor : float, optional
        The decay factor used to compute importance (default is 0.9).

    Returns
    -------
    list of float
        A list of importance weights for the elements.
    """
    indices = np.arange(n_elements)
    importance = decay_factor ** indices[::-1]
    return importance.tolist()


def get_prov_eval_metrics(labels, predicted_labels):
    """Compute evaluation metrics for provenance by comparing true and predicted labels.

    Parameters
    ----------
    labels : array-like
        True labels.
    predicted_labels : array-like
        Predicted labels.

    Returns
    -------
    dict
        A dictionary containing evaluation metrics such as "Accuracy".
    """
    accuracy = metrics.accuracy_score(labels, predicted_labels)
    answer = {
        "Accuracy": accuracy,
    }

    return answer


def normalize_contributions(contributions):
    """Normalize client contributions to sum to 1.

    Parameters
    ----------
    contributions : dict
        Dictionary mapping client IDs to their contributions.

    Returns
    -------
    dict
        Dictionary with normalized contributions.
    """
    total = sum(contributions.values())
    if total == 0:
        # If total is 0, distribute equally
        n_clients = len(contributions)
        return {client_id: 1.0 / n_clients for client_id in contributions.keys()}

    return {client_id: contrib / total for client_id, contrib in contributions.items()}
