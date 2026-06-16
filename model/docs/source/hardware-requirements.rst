Hardware Requirements
=====================

Lizzy 7B hardware needs depend on the model format, runtime, context length,
and batching. Treat the numbers below as planning guidance, then validate on
the exact runtime and prompt lengths you will use in production.

Recommended starting points
---------------------------

.. list-table::
   :header-rows: 1

   * - Use case
     - Practical starting point
     - Notes
   * - Transformers BF16, short prompts
     - 24 GB GPU memory
     - The BF16 weights are about 14 GB before runtime overhead and KV cache.
   * - Transformers BF16, long context
     - 40 GB or more GPU memory
     - Long prompts can add tens of GB of KV cache.
   * - vLLM serving
     - 24 GB or more GPU memory
     - Single-GPU A40/H100 smoke tests passed. Tensor-parallel in-process
       generation passed on 2x H100 and 4x A40; validate serving separately.
   * - GGUF Q4_K_M
     - 8 GB unified memory or RAM minimum, 16 GB recommended
     - Requires a runtime that supports the Lizzy GGUF architecture.
   * - GGUF Q5_K_M or Q6_K
     - 16 GB unified memory or RAM recommended
     - Use these when quality matters and local memory is available.
   * - GGUF Q8_0
     - 24 GB unified memory or RAM recommended
     - Near-lossless quantization needs more memory headroom.
   * - GGUF f16
     - 32 GB unified memory or RAM recommended
     - Use for local quality checks only when memory is ample.

Disk space
----------

Plan for the model file, cache duplication, and temporary download files.
The GGUF repository includes variants from about 4.5 GB for ``Q4_K_M`` to
about 14.6 GB for the f16 file. The BF16 Safetensors checkpoint requires more
disk space than the GGUF quantizations.

For a comfortable local setup, keep at least 2x the selected model size free.
This leaves room for the Hugging Face cache, partial downloads, and runtime
metadata.

Context length and KV cache
---------------------------

Long context increases memory use even after the model weights fit. Lizzy uses
32 layers, hidden size 4096, and 32 attention heads. With BF16 or FP16 KV
cache, a rough upper bound is about 0.5 MB per token for batch size 1.

Approximate KV cache sizes:

.. list-table::
   :header-rows: 1

   * - Context length
     - KV cache estimate
   * - 4,096 tokens
     - About 2 GB
   * - 8,192 tokens
     - About 4 GB
   * - 32,768 tokens
     - About 16 GB
   * - 65,536 tokens
     - About 32 GB

Batching and concurrent requests multiply KV cache use. If a runtime uses KV
cache quantization or paged attention, memory use can be lower, but validate
the actual configuration before relying on it.

CPU and memory
--------------

For GGUF local inference, prefer a modern CPU with high memory bandwidth.
Apple Silicon machines can use unified memory effectively when the runtime
supports the model architecture and can access the Metal device. In
virtualized, sandboxed, or otherwise constrained macOS environments, start with
a CPU-only smoke test before enabling GPU offload. On CPU-only systems,
generation speed depends heavily on memory bandwidth, quantization level,
thread count, and context length.

On an Apple M3 Ultra Mac Studio with the Lizzy-compatible llama.cpp branch, the
``Q4_K_M`` GGUF file offloaded all 33 layers to Metal and completed short
completion and server tests. Treat these numbers as a capability check rather
than a throughput guarantee for other prompts, quantizations, or machines.

For GPU serving, prefer CUDA-capable Linux systems for vLLM. Transformers can
also run on other accelerator backends, but deployment behaviour and memory
headroom should be tested per environment.

In short H100 vLLM smoke tests with ``vllm==0.21.0`` and
``gpu_memory_utilization=0.72``, Lizzy loaded at ``max_model_len`` values up to
32768. Tensor-parallel in-process generation passed with
``tensor_parallel_size=2`` on H100 and ``tensor_parallel_size=4`` on A40.
``tensor_parallel_size=8`` was not tested because a full 8-GPU allocation was
not available. OpenAI-compatible vLLM serving with tensor parallelism should
still be validated separately before production use.

Runtime support
---------------

Hardware is necessary but not sufficient for the GGUF files. The runtime must
also support ``general.architecture = lizzy``. If loading fails with an
architecture error, see :doc:`troubleshooting`.
