Troubleshooting
===============

Use this page when a documented runtime does not start cleanly or returns a
load-time error.

GGUF runtime reports unknown architecture
-----------------------------------------

Lizzy GGUF files report the model architecture as ``lizzy``. Local GGUF
runtimes must understand that architecture before they can load the model.

If you see an error like this, the runtime does not currently support the Lizzy
GGUF architecture:

.. code-block:: text

    unknown model architecture: 'lizzy'

The following smoke tests on macOS used the ``Q4_K_M`` GGUF file. The table
shows what happened with the tested runtime and what may become supported when
the runtime is built from, or linked against, a Lizzy-compatible llama.cpp
fork:

.. list-table::
   :header-rows: 1

   * - Runtime
     - Tested path
     - Tested result
     - With the Lizzy llama.cpp fork
   * - Stock llama.cpp 9330
     - ``llama-cli`` and ``llama-server``
     - Failed at model load with ``unknown model architecture: 'lizzy'``.
     - Use ``relogu/llama.cpp`` branch ``lorenzo-dev``. ``llama-completion``,
       the ``-hf`` shortcut, Metal offload, CPU-only inference, and
       ``llama-server`` were smoke-tested successfully at commit ``991a41b``.
   * - llama-cpp-python 0.3.23
     - ``Llama.from_pretrained``
     - Failed at model load with ``unknown model architecture: 'lizzy'``.
     - A source build of the released ``llama-cpp-python==0.3.23`` package
       with vendored ``relogu/llama.cpp`` imported successfully and began
       loading Lizzy with Metal offload. It then failed while parsing the
       embedded chat template because ``llama-cpp-python`` did not understand
       the ``generation`` Jinja tag. This path needs Python wrapper support for
       Lizzy's chat template, or a compatible template override, before it can
       be considered end-to-end supported.
   * - Ollama 0.23.4
     - ``ollama run hf.co/flwrlabs/Lizzy-7B-GGUF:Q4_K_M``
     - Downloaded the model, then returned ``unable to load model``.
     - Work in progress. Ollama needs a backend that includes Lizzy
       architecture support. This was not retested with a custom Ollama build.
   * - Desktop GGUF apps
     - LM Studio, Jan, and similar apps
     - Work in progress; not smoke-tested in this pass.
     - Check the app's llama.cpp backend version before importing Lizzy. The
       backend must recognize ``general.architecture = lizzy``. If the app
       supports custom backends, use a Lizzy-compatible llama.cpp build; if it
       reports ``unknown model architecture: 'lizzy'``, update the backend or
       run Lizzy with llama.cpp directly.

The ``lorenzo-dev`` branch of ``relogu/llama.cpp`` was smoke-tested at commit
``991a41b`` with ``lizzy-7b-q4_k_m.gguf`` and completed ``llama-completion``,
``-hf`` loading, Metal offload, CPU-only inference, and OpenAI-compatible
``llama-server`` inference. On an Apple M3 Ultra Mac Studio, the Q4_K_M file
offloaded all 33 layers to Metal and generated at about 100 tokens per second
in a short completion test. Use a Lizzy-compatible GGUF runtime or run the BF16
checkpoint with Transformers. If you maintain a local runtime, verify support
for ``general.architecture = lizzy`` before using the GGUF files in production.

To test the Lizzy-compatible llama.cpp branch directly:

.. code-block:: bash

    git clone --branch lorenzo-dev https://github.com/relogu/llama.cpp.git
    cd llama.cpp
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release --target llama-completion
    ./build/bin/llama-completion \
      -m /path/to/lizzy-7b-q4_k_m.gguf \
      -p "Q: 2+2? A:" \
      -n 16 \
      -c 1024

Metal fails to initialize on macOS
----------------------------------

In constrained macOS, virtualized, or sandboxed environments, llama.cpp can
fail before inference with an error such as:

.. code-block:: text

    ggml_metal_init: error: failed to create command queue

This is an environment or device-access issue, not the same as an unsupported
model architecture. Metal offload was verified on a real Apple M3 Ultra Mac
Studio using the Lizzy-compatible llama.cpp branch. For a low-impact CPU-only
smoke test, disable device offload and use a short context:

.. code-block:: bash

    ./build/bin/llama-completion \
      -m /path/to/lizzy-7b-q4_k_m.gguf \
      -p "Q: 2+2? A:" \
      -n 16 \
      -c 1024 \
      -t 2 \
      -dev none \
      -ngl 0 \
      --no-op-offload

Hugging Face shortcut does not use the expected cache
-----------------------------------------------------

The llama.cpp ``-hf`` shortcut was verified with the Lizzy-compatible llama.cpp
branch on a real macOS host. It can read from its own cache as well as the
Hugging Face cache. If you are testing carefully and want to avoid unexpected
downloads or writes to your home directory, set temporary cache locations:

.. code-block:: bash

    HF_HOME=/tmp/lizzy-hf-cache \
    LLAMA_CACHE=/tmp/lizzy-llama-cache \
    ./build/bin/llama-completion \
      -hf flwrlabs/Lizzy-7B-GGUF:Q4_K_M \
      -p "Q: Say hi. A:" \
      -n 8 \
      -c 1024

If offline mode reports that a small preset or metadata file is missing, rerun
without ``--offline`` once to populate the temporary llama.cpp cache, or use the
direct ``-m /path/to/lizzy-7b-q4_k_m.gguf`` form. A non-fatal ``HEAD failed,
status: 404`` message can appear during probing; the model can still resolve
and load successfully afterward.

llama-cpp-python fails while parsing the chat template
------------------------------------------------------

If a custom ``llama-cpp-python`` build gets past model loading but fails with a
Jinja error such as:

.. code-block:: text

    Encountered unknown tag 'generation'

then the native llama.cpp library is likely compatible with Lizzy, but the
Python wrapper does not support every Hugging Face chat-template extension
stored in the GGUF metadata. A temporary compatibility shim is to strip the
``generation`` and ``endgeneration`` block tags before ``llama-cpp-python``
compiles the embedded template:

.. code-block:: python

    import llama_cpp.llama_chat_format as chat_format

    original_init = chat_format.Jinja2ChatFormatter.__init__

    def lizzy_chat_template_init(
        self,
        template,
        eos_token,
        bos_token,
        add_generation_prompt=True,
        stop_token_ids=None,
    ):
        template = template.replace("{% generation %}", "")
        template = template.replace("{% endgeneration %}", "")
        return original_init(
            self,
            template,
            eos_token,
            bos_token,
            add_generation_prompt,
            stop_token_ids,
        )

    chat_format.Jinja2ChatFormatter.__init__ = lizzy_chat_template_init

Apply the shim before constructing ``Llama``. This was tested with a source
build of ``llama-cpp-python==0.3.23`` linked against ``relogu/llama.cpp`` at
commit ``991a41b``; ``create_chat_completion`` then returned the expected
response on the Q4_K_M GGUF file.

vLLM installation or serving fails locally
------------------------------------------

vLLM is intended for supported accelerator environments, commonly Linux GPU
servers. If installation or serving fails on a local laptop, test the same
command on the target GPU environment before treating it as a model issue.

On single NVIDIA A40 and H100 GPUs, ``vllm==0.21.0`` with
``torch==2.11.0+cu130`` loaded Lizzy with BF16 weights and completed short
generation tests. The A40 path used FlashAttention 2; the H100 path used
FlashAttention 3. The OpenAI-compatible vLLM server also started on H100 and
responded to a ``/v1/chat/completions`` request. vLLM used its Transformers
modeling backend for Lizzy, so validate throughput and feature coverage on the
exact serving configuration you plan to deploy.

Single-GPU H100 smoke tests passed at ``max_model_len`` values of 1024, 4096,
16384, and 32768. With ``gpu_memory_utilization=0.72``, vLLM reported about
50.7 GiB available for KV cache and maximum concurrency of about 100x at 1024
tokens, 25x at 4096 tokens, 6x at 16384 tokens, and 4x at 32768 tokens. Treat
these as short smoke-test observations, not production sizing guarantees.

Tensor-parallel in-process generation now passes in the tested environments
after Lizzy's attention implementation was updated to derive local query and
key/value head counts from the sharded projection tensors. The validated cases
are ``tensor_parallel_size=2`` on two H100 GPUs and ``tensor_parallel_size=4``
on four A40 GPUs. These tests used a fresh Hugging Face model snapshot with
``max_position_embeddings=65536`` and ``vllm==0.21.0``.

``tensor_parallel_size=8`` is expected to be the next natural configuration for
Lizzy's 32 query heads and 8 key/value heads, but it was not completed because
an 8-GPU allocation was not available during testing. OpenAI-compatible
``vllm serve`` with tensor parallelism was also not confirmed in this pass.
Validate both before relying on them in production.

Older Lizzy model snapshots can fail under vLLM tensor parallelism with an
attention reshape error such as ``shape '[1, 16384, 32, 128]' is invalid``. If
you see that error, clear the Hugging Face cache or pin to a newer model
revision that includes local tensor-parallel head-count handling in
``modeling_lizzy.py``.

On the tested H100 Slurm node, FlashInfer's sampling JIT required the
``ninja`` executable to be available on ``PATH``. Installing the Python
``ninja`` package was not enough unless the virtual environment's ``bin``
directory was also on ``PATH`` before starting vLLM.

A fully fresh Python user-base installation on the H100 node loaded the model
but failed during vLLM engine profiling because Triton could not compile a
small CUDA utility with the node's system ``gcc``. If a clean environment fails
before generation with a Triton or CUDA utility compilation error, check the
compiler, CUDA driver libraries, Python headers, and ``TMPDIR`` before treating
the failure as a Lizzy model issue.

On a two-GPU NVIDIA V100 machine, ``vllm==0.21.0`` installed successfully but
pulled ``torch==2.11.0+cu130``, whose CUDA kernels did not support V100 compute
capability 7.0. A basic CUDA tensor operation failed with ``no kernel image is
available for execution on the device``.

The same host completed a short Lizzy generation with ``vllm==0.10.2``,
``torch==2.8.0+cu128``, ``transformers==5.9.0``, ``dtype=float16``,
``max_model_len=1024``, and ``VLLM_USE_V1=0``. This path used vLLM's
Transformers fallback and XFormers backend. It also needed a temporary
compatibility shim because vLLM 0.10.2 expects the Transformers tokenizer
attribute ``all_special_tokens_extended``, while Lizzy's Transformers 5
``TokenizersBackend`` exposes ``all_special_tokens`` instead. Treat this as a
validated workaround for V100 testing, not the preferred production path.

For production vLLM serving, prefer a newer NVIDIA GPU supported by the current
vLLM and PyTorch CUDA wheels, then validate ``vllm serve`` end to end with the
exact vLLM version, GPU type, context length, and batching settings you plan to
deploy.

Transformers AutoTokenizer fails
--------------------------------

If ``AutoTokenizer.from_pretrained("flwrlabs/Lizzy-7B",
trust_remote_code=True)`` fails with:

.. code-block:: text

    Tokenizer class TokenizersBackend does not exist or is not currently imported

then your Transformers version does not include ``TokenizersBackend``. Use
Python 3.10 or later and install Transformers 5.x:

.. code-block:: bash

    pip install "transformers>=5,<6" jinja2 protobuf

``AutoTokenizer`` and ``apply_chat_template`` were tested successfully with
``transformers==5.9.0`` on Python 3.13.

Transformers generation fails with token_type_ids or cache errors
-----------------------------------------------------------------

The recommended Transformers 5.x path does not require either workaround. On
macOS with Python 3.13, ``torch==2.12.0``, and ``transformers==5.9.0``,
``AutoTokenizer`` returned only ``input_ids`` and ``attention_mask``, and
default cached generation completed successfully with the BF16 checkpoint.

If you are pinned to an older Transformers 4.x stack or using a manual
``PreTrainedTokenizerFast`` workaround, generation may fail because extra
``token_type_ids`` are passed to the model, or because cache handling raises an
error like ``'int' object has no attribute 'shape'``. In that older-stack case,
remove ``token_type_ids`` before generation and pass ``use_cache=False`` to
``generate``.

Transformers emits a RoPE scaling warning
-----------------------------------------

When loading the Transformers config, the current model metadata can emit a
warning about the explicit RoPE factor differing from the implicit factor. This
warning comes from the model configuration. Validate generation quality on your
target runtime and pin the model revision used for deployment.

Downloads are large
-------------------

The smallest GGUF variant is several gigabytes, and the BF16 checkpoint is
larger. Check disk space and network stability before running the examples.
For constrained machines, start with ``Q4_K_M`` once your runtime supports
Lizzy GGUF. See :doc:`hardware-requirements` for memory and disk planning
guidance.

The curl example cannot connect
-------------------------------

The curl example expects an OpenAI-compatible server listening on
``localhost:8000`` and serving the model name ``flwrlabs/Lizzy-7B``. Start the
server first, confirm the port, and keep the model name in the request body
aligned with the served model.
