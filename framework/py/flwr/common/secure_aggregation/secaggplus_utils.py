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
    """Seeded pseudo-random number generator for noise generation with Numpy."""
    assert len(seed) & 0x3 == 0
    seed32 = 0
    for i in range(0, len(seed), 4):
        seed32 ^= int.from_bytes(seed[i : i + 4], "little")
    # pylint: disable-next=no-member
    gen = np.random.RandomState(seed32)
    output = []
    for dimension in dimensions_list:
        if len(dimension) == 0:
            arr = np.array(gen.randint(0, num_range - 1), dtype=np.int64)
        else:
            arr = gen.randint(0, num_range - 1, dimension, dtype=np.int64)
        output.append(arr)
    return output
