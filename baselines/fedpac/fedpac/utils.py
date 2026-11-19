"""fedpac: A Flower Baseline."""
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from flwr.common import NDArrays
import cvxpy as cvx
from functools import reduce


def pairwise(data):
    """Generate pairs (x, y) in a tuple such that index x < index y.

    Args:
    data Indexable (including ability to query length) containing the elements
    Returns:
    Generator over the pairs of the elements of 'data'
    """
    n = len(data)
    for i in range(n):
        for j in range(i, n):
            yield (data[i], data[j])

def get_centroid(feature_list: List[torch.FloatTensor]):
    """Take in the feature extraction layers list and returns mean of feature extraction
    layers of each labels.
    """
    features_mean = {}
    for [label, feat_list] in feature_list.items():
        feat = 0 * feat_list[0]
        for i in feat_list:
            feat += i
        features_mean[label] = feat / len(feat_list)

    return features_mean


def aggregate_weights(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def aggregate_centroids(centroids_list, class_sizes):
    """Compute estimated global centroid."""
    aggregated_centroids_list, aggregated_class_sizes = {}, {}
    for i in range(len(centroids_list)):
        for label, centroid in centroids_list[i].items():
            if label in aggregated_centroids_list.keys():
                aggregated_centroids_list[label].append(centroid)
                aggregated_class_sizes[label].append(class_sizes[i][label])
            else:
                aggregated_centroids_list[label] = [centroid]
                aggregated_class_sizes[label] = [class_sizes[i][label]]

    for label, centroid in aggregated_centroids_list.items():
        size_list = aggregated_class_sizes[label]
        c = torch.zeros(len(centroid[0]))
        for i in range(len(centroid)):
            c += centroid[i] * size_list[i]
        aggregated_centroids_list[label] = c / sum(size_list).item()

    return aggregated_centroids_list


def aggregate_heads(stats, device):
    var = [s[0] for s in stats]
    bias = [s[1] for s in stats]

    num_clients = len(var)
    num_class = bias[0].shape[0]
    d = bias[0].shape[1]
    agg_head = []
    bias = torch.stack(bias).to(device)

    for i in range(num_clients):
        v = torch.tensor(var, device=device)
        href = bias[i]
        dist = torch.zeros((num_clients, num_clients), device=device)

        for j, k in pairwise(tuple(range(num_clients))):
            hj = bias[j]
            hk = bias[k]
            h = torch.zeros((d, d), device=device)
            for m in range(num_class):
                h += torch.mm(
                    (href[m] - hj[m]).reshape(d, 1), (href[m] - hk[m]).reshape(1, d)
                )

            d_jk = torch.trace(h)
            dist[j][k] = d_jk
            dist[k][j] = d_jk

        p_matrix = torch.diag(v) + dist
        evals, evecs = torch.linalg.eig(p_matrix)

        p_matrix = p_matrix.cpu().numpy()  # coefficient for QP problem

        p_matrix_new = 0
        for x in range(num_clients):
            if evals.real[x] >= 0.01:
                p_matrix_new += evals[x] * torch.mm(
                    evecs[:, x].reshape(num_clients, 1),
                    evecs[:, x].reshape(1, num_clients),
                )
                p_matrix = (
                    p_matrix_new.cpu().numpy()
                    if not np.all(np.linalg.eigvals(p_matrix) >= 0.0)
                    else p_matrix
                )

        alpha = 0
        eps = 1e-3

        if np.all(np.linalg.eigvals(p_matrix) >= 0):
            alphav = cvx.Variable(num_clients)
            obj = cvx.Minimize(cvx.quad_form(alphav, p_matrix))
            prob = cvx.Problem(obj, [cvx.sum(alphav) == 1.0, alphav >= 0])
            prob.solve()
            alpha = alphav.value
            alpha = [(a) * (a > eps) for a in alpha]  # zero-out small weights (<eps)

        else:
            alpha = None

        agg_head.append(alpha)
    return agg_head
