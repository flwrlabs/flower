Compress model updates with TurboQuant
=====================================

Flower compression pipelines provide a Message API mechanism for reducing model
communication without changing strategy or ClientApp training code. The API is
inspired by OpenFL pipelines: ``forward`` compresses an array and ``backward``
decompresses it.

Overview
--------

Compression is represented as a Flower-native ``Array`` serialization type:
``flwr.compressed_pipeline``. A compressed Array stores an envelope containing:

* ``version``: envelope format version
* ``pipeline_id``: for example ``turboquant_mse``
* ``pipeline_params``: bit width, block size, and related parameters
* ``metadata``: transformer metadata, such as shape, dtype, scales, and payload sizes
* ``payload``: compressed bytes

``Array.numpy()`` decodes compressed Arrays automatically, so most strategy and
ClientApp code continues to consume NumPy arrays normally.

TurboQuant MSE
--------------

``TurboQuantMSEPipeline`` is a block-normalized lossy quantizer designed for
large model deltas. Its randomized rotation follows the concentration step in
the `TurboQuant paper <https://arxiv.org/abs/2504.19874>`_.

For each array:

1. Flatten the array.
2. Split it into fixed-size blocks.
3. Apply a deterministic randomized Walsh-Hadamard rotation, which spreads
   outliers and makes block coordinates close to normally distributed.
4. Compute one RMS scale per rotated block.
5. Normalize each rotated block by its scale.
6. Quantize normalized values with a Lloyd-Max-style normal codebook.
7. Pack centroid indices into ``n`` bits.
8. Store packed indices plus fp16 block scales in the compressed payload.

On decode, the centroid indices are unpacked, multiplied by their block scales,
inverse-rotated, reshaped, and cast back to the original dtype. Rotation requires
a power-of-two block size and is enabled by default.

Delta compression
-----------------

For federated learning, compressing full model weights is often unnecessary.
``DeltaState`` provides a small stateful helper for delta workflows:

.. code-block:: python

    from flwr.common.compression import DeltaState

    state = DeltaState.from_arrayrecord(global_arrays)
    delta = state.extract_delta(client_arrays)
    restored = state.apply_delta(delta)

The intended bidirectional pattern is:

* Server sends compressed global deltas to clients.
* Clients apply the delta to their local reference model.
* Clients train locally.
* Clients send compressed local deltas back to the server.
* Server decodes and aggregates deltas layer-wise.

ClientApp integration
---------------------

Use ``CompressionMod`` to compress outgoing client replies:

.. code-block:: python

    from flwr.clientapp.mod import CompressionMod

    app = ClientApp(
        mods=[CompressionMod("turboquant_mse", n_bits=3, block_size=262_144)]
    )

The mod also reads per-message config keys:

* ``compression-pipeline``
* ``compression-n-bits``
* ``compression-block-size``

ServerApp integration
---------------------

Wrap a message-based strategy to compress outbound server messages:

.. code-block:: python

    from flwr.serverapp.strategy import CompressionStrategy, FedAvg

    strategy = CompressionStrategy(
        FedAvg(),
        "turboquant_mse",
        n_bits=3,
        block_size=262_144,
    )

CUDA acceleration
-----------------

The optional CUDA path is valuable for 1B+ and 70B-scale models because CPU-side
compression becomes the bottleneck. Enable it with ``use_cuda=True`` when
constructing ``TurboQuantMSEPipeline`` or, in ``flowertune-llm``, with:

.. code-block:: shell

    flwr run . --run-config "compression.enabled=true compression.n-bits=3 compression.cuda-enabled=true"

The current CUDA implementation is used for 3- and 4-bit TurboQuant MSE. Other
bit widths automatically fall back to the CPU path. The CUDA path keeps model deltas
on device and streams them in chunks:

1. Generate or receive a CUDA delta tensor.
2. Quantize with CUDA tensor operations.
3. Pack 3- or 4-bit centroid IDs on CUDA.
4. Copy only the compressed payload and fp16 scales to host/network buffers.
5. Copy compressed payloads back to CUDA on receive.
6. Unpack and dequantize on CUDA.

This avoids materializing a full 70B model on CPU and avoids transferring raw
bf16 deltas across the PCIe boundary.

In the 70B-scale benchmark on an RTX 5090, a full streamed 68.98B-parameter pass
with 3-bit packing, unpacking, and compressed payload host/device transfer took
about 10.25 seconds per full model pass. The raw bf16 payload was 137.95 GB and
the compressed payload was 25.87 GB, a 5.33x reduction.

Current limitations
-------------------

* CUDA acceleration currently targets the 3- and 4-bit packed TurboQuant MSE paths.
* The implementation uses PyTorch CUDA tensor operations. A future production
  optimization should replace the Python CUDA tensor implementation with a fused
  CUDA or Triton extension.
* Error feedback and compressed-domain aggregation are separate follow-up features.
