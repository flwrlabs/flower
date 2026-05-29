Lizzy GGUF
==========

`flwrlabs/Lizzy-7B-GGUF <https://huggingface.co/flwrlabs/Lizzy-7B-GGUF>`_
contains quantized GGUF versions of Lizzy 7B for efficient local inference.
Use these files when you want CPU inference, flexible GPU offload, smaller
model files, or fast loading with a runtime that supports the Lizzy GGUF
architecture.

Available variants
------------------

.. list-table::
   :header-rows: 1

   * - Variant
     - Reported file size
     - Reported quality retention
     - Recommended use
   * - Q4_K_M
     - 4.2 GB
     - 92%
     - Resource-constrained environments
   * - Q5_K_M
     - 4.8 GB
     - 95%
     - Best balance of quality and size
   * - Q6_K
     - 5.6 GB
     - 97%
     - Between Q5 and Q8
   * - Q8_0
     - 7.2 GB
     - 99%
     - Near-lossless compression
   * - f16
     - 13.6 GB
     - 100%
     - Maximum quality and benchmarking

The Hugging Face file browser can show slightly different sizes because
rounded sizes and file-browser metadata use different views.

All listed GGUF files were smoke-tested with the Lizzy-compatible
``relogu/llama.cpp`` branch at commit ``991a41b`` using CPU-only inference,
``n_ctx=512``, and a short prompt. The quantized files loaded through the
llama.cpp Hugging Face shortcut using ``Q4_K_M``, ``Q5_K_M``, ``Q6_K``, and
``Q8_0`` selectors. The f16 file, ``lizzy-final.gguf``, was tested by
downloading the file directly and loading it with ``-m``.

Recommended default
-------------------

Start with ``Q5_K_M`` when quality matters and you still want a compact local
model. Use ``Q4_K_M`` when memory or disk is tight. Use ``Q8_0`` or ``f16``
for quality-sensitive benchmarking or when local resources are less
constrained. See :doc:`hardware-requirements` for local memory and disk
planning guidance.

Architecture details
--------------------

The GGUF release reports:

- base model: Lizzy 7B
- layers: 32 with post-norm architecture
- hidden size: 4096
- attention: sliding window 4096 plus full attention
- RoPE: YaRN scaling with factor 8.0 and original context 8192
- vocabulary: 100,278 tokens
- context: 65,536 tokens
- tensors: 355, including ``attn_post_norm`` and ``ffn_post_norm``

Reasoning behaviour
-------------------

Lizzy 7B can emit reasoning tokens before the final answer. Applications that
expose model output directly should decide whether to show, hide, or
post-process reasoning traces according to product and safety requirements.

When to use GGUF
----------------

Use GGUF when:

- you need CPU inference
- you want flexible GPU layer offload
- you need smaller model files
- you are using a local runtime that supports the Lizzy GGUF architecture
- fast local loading matters

Use the original BF16 Safetensors checkpoint when:

- you need full precision
- you are using Transformers or vLLM
- you need serving features not available in your GGUF runtime
- you want to fine-tune the model

Example
-------

The example below requires a llama.cpp build that supports the Lizzy GGUF
architecture. The ``lorenzo-dev`` branch of ``relogu/llama.cpp`` is the
currently tested compatibility branch for Lizzy. Use upstream llama.cpp or a
packaged runtime instead once it includes support for
``general.architecture = lizzy``. If another runtime reports
``unknown model architecture: 'lizzy'``, see :doc:`troubleshooting`.

.. code-block:: bash

    git clone --branch lorenzo-dev https://github.com/relogu/llama.cpp.git
    cd llama.cpp
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release --target llama-server
    ./build/bin/llama-server \
      -hf flwrlabs/Lizzy-7B-GGUF:Q5_K_M \
      -c 1024

Then use the llama.cpp web UI or OpenAI-compatible local API exposed by
``llama-server``.
