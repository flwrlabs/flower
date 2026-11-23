# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Workflow for the SecAgg protocol."""


from .secaggplus_workflow import SecAggPlusWorkflow


class SecAggWorkflow(SecAggPlusWorkflow):
    """The workflow for the SecAgg protocol.

    The SecAgg protocol ensures the secure summation of integer vectors owned by
    multiple parties, without accessing any individual integer vector. This workflow
    allows the server to compute the weighted average of model parameters across all
    clients, ensuring individual contributions remain private. This is achieved by
    clients sending both, a weighting factor and a weighted version of the locally
    updated parameters, both of which are masked for privacy. Specifically, each
    client uploads "[w, w * params]" with masks, where weighting factor 'w' is the
    number of examples ('num_examples') and 'params' represents the model parameters
    ('parameters') from the client's `FitRes`. The server then aggregates these
    contributions to compute the weighted average of model parameters.

    Note that only this uploaded vector is masked. Each client's 'num_examples' and
    'metrics' from its `FitRes` are also sent to the server unmasked and are passed
    to `Strategy.aggregate_fit` per client.

    The protocol involves four main stages:

    - 'setup': Send SecAgg configuration to clients and collect their public keys.
    - 'share keys': Broadcast public keys among clients and collect encrypted secret
      key shares.
    - 'collect masked vectors': Forward encrypted secret key shares to target clients
      and collect masked model parameters.
    - 'unmask': Collect secret key shares to decrypt and aggregate the model parameters.

    Only the aggregated model parameters are exposed and passed to
    `Strategy.aggregate_fit`, ensuring individual data privacy.

    Parameters
    ----------
    reconstruction_threshold : Union[int, float]
        The minimum number of shares required to reconstruct a client's private key,
        or, if specified as a float, it represents the proportion of the total number
        of shares needed for reconstruction. This threshold ensures privacy by allowing
        for the recovery of contributions from dropped clients during aggregation,
        without compromising individual client data.
    max_weight : Optional[float] (default: 1000.0)
        The maximum value of the weight that can be assigned to any single client's
        update during the weighted average calculation on the server side, e.g., in the
        FedAvg algorithm.
    clipping_range : float, optional (default: 8.0)
        The range within which model parameters are clipped before quantization.
        This parameter ensures each model parameter is bounded within
        [-clipping_range, clipping_range], facilitating quantization.
    quantization_range : int, optional (default: 4194304, this equals 2**22)
        The size of the range into which floating-point model parameters are quantized,
        mapping each parameter to an integer in [0, quantization_range-1]. This
        facilitates cryptographic operations on the model updates.
    modulus_range : int, optional (default: 4294967296, this equals 2**32)
        The range of values from which random mask entries are uniformly sampled
        ([0, modulus_range-1]). `modulus_range` must be less than 4294967296.
        Please use 2**n values for `modulus_range` to prevent overflow issues.
    timeout : Optional[float] (default: None)
        The timeout duration in seconds. If specified, the workflow will wait for
        replies for this duration each time. If `None`, there is no time limit and
        the workflow will wait until replies for all messages are received.

    Notes
    -----
    - Each client's private key is split into N shares under the SecAgg protocol, where
      N is the number of selected clients.
    - Generally, higher `reconstruction_threshold` means better privacy guarantees but
      less tolerance to dropouts.
    - Too large `max_weight` may compromise the precision of the quantization.
    - `modulus_range` must be 2**n and larger than `quantization_range`.
    - When `reconstruction_threshold` is a float, it is interpreted as the proportion of
      the number of all selected clients needed for the reconstruction of a private key.
      This feature enables flexibility in setting the security threshold relative to the
      number of selected clients.
    - `reconstruction_threshold`, and the quantization parameters
      (`clipping_range`, `quantization_range`, `modulus_range`) play critical roles in
      balancing privacy, robustness, and efficiency within the SecAgg protocol.
    """

    def __init__(  # pylint: disable=R0913
        self,
        reconstruction_threshold: int | float,
        *,
        max_weight: float = 1000.0,
        clipping_range: float = 8.0,
        quantization_range: int = 4194304,
        modulus_range: int = 4294967296,
        timeout: float | None = None,
    ) -> None:
        super().__init__(
            num_shares=1.0,
            reconstruction_threshold=reconstruction_threshold,
            max_weight=max_weight,
            clipping_range=clipping_range,
            quantization_range=quantization_range,
            modulus_range=modulus_range,
            timeout=timeout,
        )
