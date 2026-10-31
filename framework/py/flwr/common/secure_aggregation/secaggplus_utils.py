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
"""Utility functions for the SecAgg/SecAgg+ protocol."""


import hashlib
import hmac
import struct

import numpy as np

from flwr.common import NDArrayInt


def share_keys_plaintext_concat(
    src_node_id: int, dst_node_id: int, b_share: bytes, sk_share: bytes
) -> bytes:
    """Combine arguments to bytes.

    Parameters
    ----------
    src_node_id : int
        the node ID of the source.
    dst_node_id : int
        the node ID of the destination.
    b_share : bytes
        the private key share of the source sent to the destination.
    sk_share : bytes
        the secret key share of the source sent to the destination.

    Returns
    -------
    bytes
        The combined bytes of all the arguments.
    """
    return b"".join(
        [
            int.to_bytes(src_node_id, 8, "little", signed=False),
            int.to_bytes(dst_node_id, 8, "little", signed=False),
            int.to_bytes(len(b_share), 4, "little"),
            b_share,
            sk_share,
        ]
    )


def share_keys_plaintext_separate(plaintext: bytes) -> tuple[int, int, bytes, bytes]:
    """Retrieve arguments from bytes.

    Parameters
    ----------
    plaintext : bytes
        the bytes containing 4 arguments.

    Returns
    -------
    src_node_id : int
        the node ID of the source.
    dst_node_id : int
        the node ID of the destination.
    b_share : bytes
        the private key share of the source sent to the destination.
    sk_share : bytes
        the secret key share of the source sent to the destination.
    """
    src, dst, mark = (
        int.from_bytes(plaintext[:8], "little", signed=False),
        int.from_bytes(plaintext[8:16], "little", signed=False),
        int.from_bytes(plaintext[16:20], "little"),
    )
    ret = (src, dst, plaintext[20 : 20 + mark], plaintext[20 + mark :])
    return ret


def pseudo_rand_gen(
    seed: bytes, num_range: int, dimensions_list: list[tuple[int, ...]]
) -> list[NDArrayInt]:
    """Seeded pseudo-random number generator for noise generation.

    Uses HMAC-SHA256 in counter mode to generate a cryptographically strong,
    deterministic byte stream from the seed, preserving full entropy.

    Assumes `num_range` is a power of two and >= 2.
    """
    if num_range < 2 or (num_range & (num_range - 1)) != 0:
        raise ValueError("num_range must be a power of two and >= 2.")
    if num_range > (1 << 63):
        raise ValueError("num_range must be <= 2**63 to fit into int64 masks.")

    num_bytes = (num_range.bit_length() + 6) // 8
    bitmask = num_range - 1

    counter = 0
    stream_buffer = bytearray()
    masks = []

    for shape in dimensions_list:
        total_elements = int(np.prod(shape)) if shape else 1
        tensor_bytes = total_elements * num_bytes

        while len(stream_buffer) < tensor_bytes:
            h = hmac.new(seed, struct.pack("<Q", counter), hashlib.sha256)
            stream_buffer.extend(h.digest())
            counter += 1

        tensor_buffer = stream_buffer[:tensor_bytes]
        stream_buffer = stream_buffer[tensor_bytes:]

        if num_bytes == 1:
            flat_vals = np.frombuffer(tensor_buffer, dtype=np.uint8).astype(np.int64)
        elif num_bytes == 2:
            flat_vals = np.frombuffer(tensor_buffer, dtype=">u2").astype(np.int64)
        elif num_bytes == 4:
            flat_vals = np.frombuffer(tensor_buffer, dtype=">u4").astype(np.int64)
        elif num_bytes == 8:
            flat_vals = np.frombuffer(tensor_buffer, dtype=">u8")
            flat_vals = (flat_vals & bitmask).astype(np.int64)
        else:
            raw_bytes = np.frombuffer(tensor_buffer, dtype=np.uint8).reshape(-1, num_bytes)
            flat_vals = np.zeros(total_elements, dtype=np.int64)
            for i in range(num_bytes):
                flat_vals = (flat_vals << 8) | raw_bytes[:, i]

        if num_bytes != 8:
            flat_vals = flat_vals & bitmask

        if not shape:
            masks.append(np.array(flat_vals[0], dtype=np.int64))
        else:
            masks.append(flat_vals.reshape(shape))

    return masks
