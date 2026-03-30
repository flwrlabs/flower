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
"""Local DP modifier."""


from logging import INFO

import numpy as np

from flwr.clientapp.typing import ClientAppCallable
from flwr.common import Array, ArrayRecord
from flwr.common.context import Context
from flwr.common.differential_privacy import (
    add_gaussian_noise_inplace,
    compute_clip_model_update,
)
from flwr.common.logger import log
from flwr.common.message import Message

from .centraldp_mods import _handle_array_key_mismatch_err, _handle_multi_record_err


class LocalDpMod:
    """Modifier for local differential privacy.

    This mod clips the client model updates and
    adds noise to the params before sending them to the server.

    It operates on messages of type `MessageType.TRAIN`.

    Parameters
    ----------
    clipping_norm : float
        The value of the clipping norm.
    sensitivity : float
        The sensitivity of the client model.
    epsilon : float
        The privacy budget.
        Smaller value of epsilon indicates a higher level of privacy protection.
    delta : float
        The failure probability.
        The probability that the privacy mechanism
        fails to provide the desired level of privacy.
        A smaller value of delta indicates a stricter privacy guarantee.

    Examples
    --------
    Create an instance of the local DP mod and add it to the client-side mods::

        local_dp_mod = LocalDpMod( ... )
        app = fl.client.ClientApp(mods=[local_dp_mod])
    """

    def __init__(
        self, clipping_norm: float, sensitivity: float, epsilon: float, delta: float
    ) -> None:
        if clipping_norm <= 0:
            raise ValueError("The clipping norm should be a positive value.")

        if sensitivity < 0:
            raise ValueError("The sensitivity should be a non-negative value.")

        if epsilon < 0:
            raise ValueError("Epsilon should be a non-negative value.")

        if delta < 0:
            raise ValueError("Delta should be a non-negative value.")

        self.clipping_norm = clipping_norm
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = delta

    def __call__(
        self, msg: Message, ctxt: Context, call_next: ClientAppCallable
    ) -> Message:
        """Perform local DP on the client model parameters.

        Parameters
        ----------
        msg : Message
            The message received from the ServerApp.
        ctxt : Context
            The context of the ClientApp.
        call_next : ClientAppCallable
            The callable to call the next mod (or the ClientApp) in the chain.

        Returns
        -------
        Message
            The modified message to be sent back to the server.
        """
        if len(msg.content.array_records) != 1:
            return _handle_multi_record_err("LocalDpMod", msg, ArrayRecord)

        # Record array record communicated to client and clipping norm
        original_array_record = next(iter(msg.content.array_records.values()))

        # Call inner app
        out_msg = call_next(msg, ctxt)

        # Check if the msg has error
        if out_msg.has_error():
            return out_msg

        # Ensure reply has a single ArrayRecord
        if len(out_msg.content.array_records) != 1:
            return _handle_multi_record_err("LocalDpMod", out_msg, ArrayRecord)

        new_array_record_key, client_to_server_arrecord = next(
            iter(out_msg.content.array_records.items())
        )

        # Ensure keys in returned ArrayRecord match those in the one sent from server
        if list(original_array_record.keys()) != list(client_to_server_arrecord.keys()):
            return _handle_array_key_mismatch_err("LocalDpMod", out_msg)

        client_to_server_ndarrays = client_to_server_arrecord.to_numpy_ndarrays()

        # Clip the client update
        compute_clip_model_update(
            client_to_server_ndarrays,
            original_array_record.to_numpy_ndarrays(),
            self.clipping_norm,
        )
        log(
            INFO,
            "LocalDpMod: parameters are clipped by value: %.4f.",
            self.clipping_norm,
        )

        std_dev = (
            self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        )
        add_gaussian_noise_inplace(
            client_to_server_ndarrays,
            std_dev,
        )
        log(
            INFO,
            "LocalDpMod: local DP noise with %.4f stddev added to parameters",
            std_dev,
        )

        # Replace outgoing ArrayRecord's Array while preserving their keys
        out_msg.content[new_array_record_key] = ArrayRecord(
            {
                k: Array(np.asarray(v))
                for k, v in zip(
                    client_to_server_arrecord.keys(),
                    client_to_server_ndarrays,
                    strict=True,
                )
            }
        )
        return out_msg
