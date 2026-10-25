Run Lizzy
=========

This guide shows the most common ways to run Lizzy 7B from Hugging Face.
Use the BF16 checkpoint when you want full precision, Transformers/vLLM
serving, tensor parallelism, or fine-tuning. Use GGUF when you have a local
runtime that supports the Lizzy GGUF architecture and want smaller files or
local inference.

Before choosing a runtime, check :doc:`hardware-requirements` for memory,
disk, and long-context planning guidance.

Run with Transformers
---------------------

Use Python 3.10 or later and install PyTorch, Transformers 5.x, ``jinja2``,
and ``protobuf``. Transformers 5.x includes the ``TokenizersBackend`` tokenizer
class used by the Lizzy tokenizer metadata:

.. code-block:: bash

    pip install "transformers>=5,<6" torch accelerate jinja2 protobuf

Then load the base checkpoint with ``trust_remote_code=True``:

.. code-block:: python

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo_id = "flwrlabs/Lizzy-7B"

    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": "You are Lizzy 7B."},
        {"role": "user", "content": "Summarise why queue etiquette matters in the UK."},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
    )
    response = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )
    print(response)

This path was tested on macOS with Python 3.13, ``torch==2.12.0``,
``transformers==5.9.0``, and the BF16 checkpoint. ``AutoTokenizer``,
chat-template rendering, and default cached generation returned the expected
short response.

Run with vLLM
-------------

vLLM exposes an OpenAI-compatible API for GPU serving. Use this path on a
Linux GPU server or another environment supported by your vLLM version:

.. code-block:: bash

    pip install vllm
    vllm serve "flwrlabs/Lizzy-7B" --trust-remote-code

Then create ``request.json``:

.. code-block:: json

    {
      "model": "flwrlabs/Lizzy-7B",
      "messages": [
        {
          "role": "user",
          "content": "What is the capital of France?"
        }
      ]
    }

Then call the server:

.. code-block:: bash

    curl -X POST "http://localhost:8000/v1/chat/completions" \
      -H "Content-Type: application/json" \
      --data @request.json

This path was smoke-tested with ``vllm==0.21.0`` and
``torch==2.11.0+cu130`` on single NVIDIA A40 and H100 GPUs. The
OpenAI-compatible server responded on H100, and in-process generation worked
on H100 up to ``max_model_len=32768`` in short tests.

Tensor-parallel in-process generation also passed with ``tensor_parallel_size=2``
on two H100 GPUs and ``tensor_parallel_size=4`` on four A40 GPUs after Lizzy's
attention reshape logic was updated to use local tensor-parallel head counts.
``tensor_parallel_size=8`` and OpenAI-compatible serving with tensor
parallelism were not completed in this test pass, so validate the exact vLLM
server configuration before relying on multi-GPU serving in production. Runtime
support can vary by vLLM version, GPU generation, and backend.

Run the GGUF model with llama.cpp
---------------------------------

The GGUF release provides quantized files for local runtimes that support the
Lizzy GGUF architecture. The Hugging Face quick start uses the ``Q4_K_M``
variant:

The command below uses the currently tested Lizzy-compatible llama.cpp fork and
branch. Use upstream llama.cpp or a packaged runtime instead once it includes
support for ``general.architecture = lizzy``.

.. code-block:: bash

    git clone --branch lorenzo-dev https://github.com/relogu/llama.cpp.git
    cd llama.cpp
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release --target llama-completion llama-server
    ./build/bin/llama-server \
      -hf flwrlabs/Lizzy-7B-GGUF:Q4_K_M \
      -c 1024

If your runtime reports ``unknown model architecture: 'lizzy'``, it does not
include Lizzy GGUF support. Use a Lizzy-compatible build or run the BF16
checkpoint with Transformers instead. See :doc:`troubleshooting` for details
from smoke tests.

For direct terminal completion:

.. code-block:: bash

    ./build/bin/llama-completion \
      -hf flwrlabs/Lizzy-7B-GGUF:Q4_K_M \
      -p "Q: 2+2? A:" \
      -n 16 \
      -c 1024

In constrained macOS, virtualized, or sandboxed environments, Metal device
initialization can fail. To force a small CPU-only smoke test, use:

.. code-block:: bash

    ./build/bin/llama-completion \
      -hf flwrlabs/Lizzy-7B-GGUF:Q4_K_M \
      -p "Q: 2+2? A:" \
      -n 16 \
      -c 1024 \
      -t 2 \
      -dev none \
      -ngl 0 \
      --no-op-offload

Run the GGUF model with llama-cpp-python
----------------------------------------

This path requires a llama-cpp-python build linked against a llama.cpp version
that supports the Lizzy GGUF architecture. It may also require
llama-cpp-python support for Lizzy's embedded chat template. If initialization
fails with an unknown Jinja ``generation`` tag, use the llama.cpp server path
or see :doc:`troubleshooting`.

.. code-block:: python

    import llama_cpp.llama_chat_format as chat_format
    from llama_cpp import Llama

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

    llm = Llama(
        model_path="/path/to/lizzy-7b-q4_k_m.gguf",
        n_ctx=1024,
        n_gpu_layers=-1,
    )

    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": "Reply with exactly: ok"},
        ],
        max_tokens=16,
        temperature=0,
    )
    print(output["choices"][0]["message"]["content"])

Run the GGUF model with Ollama
------------------------------

Ollama support is work in progress. Current public Ollama builds may not yet
include backend support for the Lizzy GGUF architecture. Until that support is
available, prefer the Lizzy-compatible llama.cpp path above.

.. code-block:: bash

    ollama run hf.co/flwrlabs/Lizzy-7B-GGUF:Q4_K_M

Desktop GGUF apps such as LM Studio and Jan are also work in progress for
Lizzy. To use Lizzy in a desktop app, first confirm that the app version ships
with a llama.cpp backend that recognizes ``general.architecture = lizzy``. If
the app lets you choose a custom backend, point it at a Lizzy-compatible
llama.cpp build and import one of the ``flwrlabs/Lizzy-7B-GGUF``
quantizations. If it reports ``unknown model architecture: 'lizzy'``, update
the app backend or use llama.cpp directly.

Choose a runtime
----------------

.. list-table::
   :header-rows: 1

   * - Runtime
     - Use it when
   * - Transformers
     - You need Python integration, full precision, custom model code, or
       fine-tuning workflows.
   * - vLLM
     - You need GPU serving, OpenAI-compatible APIs, batching, or tensor
       parallelism.
   * - llama.cpp
     - You have a build that supports Lizzy GGUF and need local inference, CPU
       support, flexible GPU offload, or a small deployment footprint.
   * - Ollama, LM Studio, Jan
     - Work in progress. Use only after verifying that the app backend includes
       Lizzy GGUF support.

Generation settings
-------------------

The GGUF examples use ``temperature=0.6`` and ``top_p=0.95``.
For deterministic documentation, coding, or extraction tasks, start with a
lower temperature such as ``0.2``. For more conversational output, increase the
temperature gradually and evaluate outputs for factuality and style.
