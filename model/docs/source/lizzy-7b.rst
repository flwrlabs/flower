Lizzy 7B
========

Lizzy 7B is an open-weight Flower Labs language model. It is designed for
general assistant use, reasoning, coding assistance, and UK-oriented language
and knowledge. The model is available on Hugging Face in two formats:

- `flwrlabs/Lizzy-7B <https://huggingface.co/flwrlabs/Lizzy-7B>`_: the
  original BF16 Safetensors checkpoint for Transformers, vLLM, SGLang, and
  other GPU-serving stacks.
- `flwrlabs/Lizzy-7B-GGUF <https://huggingface.co/flwrlabs/Lizzy-7B-GGUF>`_:
  GGUF quantizations for local inference with runtimes that support the Lizzy
  GGUF architecture. The ``lorenzo-dev`` branch of ``relogu/llama.cpp`` has
  been smoke-tested with the Q4_K_M file.

At a glance
-----------

.. list-table::
   :header-rows: 1

   * - Property
     - Value
   * - Publisher
     - Flower Labs
   * - Model family
     - Lizzy
   * - Parameter scale
     - 7B-class
   * - Architecture
     - Decoder-only transformer
   * - Context length
     - Up to 65,536 tokens, depending on runtime and serving profile
   * - Primary language
     - English, with British English and UK-oriented behaviour enhancements
   * - Original checkpoint
     - BF16 Safetensors
   * - Quantized checkpoint
     - GGUF variants including Q4_K_M, Q5_K_M, Q6_K, Q8_0, and f16
   * - License
     - Apache-2.0 for the base model; GGUF redistribution terms refer back to
       the base model license

Architecture and configuration
------------------------------

Lizzy 7B is a 32-layer decoder-only transformer with long-context support.
The release uses 32 attention heads, sliding/local attention behaviour,
custom chat/control tokens, and deployment-specific serving configuration.

The GGUF release reports the following serving-oriented configuration:

- 32 layers with post-norm architecture
- hidden size 4096
- sliding-window attention with a 4096-token window plus full-attention
  behaviour
- YaRN RoPE scaling with factor 8.0 and original context 8192
- 100,278-token vocabulary
- 65,536-token context

Training approach
-----------------

Lizzy 7B was produced through a multi-stage training process:

- pre-training on large-scale public text, document, code, math, and
  encyclopedic corpora
- supervised fine-tuning on instruction-following, dialogue, reasoning, and
  tool-use examples
- direct preference optimisation for helpfulness, style, and answer quality
- reinforcement learning with verifiable rewards for targeted behavioural
  refinement

Training data sources include broad public text and knowledge sources,
instruction and preference data, and UK-specific examples and preference
signals.

Evaluation highlights
---------------------

The release compares Lizzy 7B with EuroLLM 9B and Apertus 8B on UK-oriented
benchmarks and broader public benchmarks.

.. list-table::
   :header-rows: 1

   * - Benchmark
     - Lizzy 7B
     - EuroLLM 9B
     - Apertus 8B
   * - Britishness MCQ
     - 71.0
     - 77.6
     - 80.8
   * - Britishness CoT
     - 80.1
     - 72.1
     - 31.7
   * - Britishness Domains
     - 89.9
     - 69.0
     - 32.6

.. list-table::
   :header-rows: 1

   * - Benchmark
     - Lizzy 7B
     - EuroLLM 9B
     - Apertus 8B
   * - MATH
     - 77.9
     - 31.3
     - 22.4
   * - MMLU
     - 67.9
     - 57.4
     - 63.4
   * - GPQA
     - 34.6
     - 26.8
     - 28.1
   * - HumanEvalPlus
     - 70.2
     - 28.2
     - 33.4
   * - MBPP+
     - 52.5
     - 41.7
     - 42.3
   * - LiveCodeBench v3
     - 39.1
     - 6.3
     - 8.5
   * - AIME
     - 35.8
     - 0.2
     - 0.6
   * - GSM8K
     - 91.8
     - 64.7
     - 64.7

Lizzy 7B trails the comparison set on Britishness MCQ recall-style probing,
but leads on Britishness CoT, Britishness domain reasoning, and most listed
reasoning, math, knowledge, and coding benchmarks.

Safety and limitations
----------------------

Lizzy 7B should be treated as an assistant model that can make mistakes. It
can produce incorrect, outdated, or over-confident responses, and higher-risk
workflows require human oversight, domain review, and downstream moderation.
The UK-oriented tuning improves local style and cultural alignment, but it can
also bias tone and assumptions toward UK conventions.

The safety-evaluation summary reports:

.. list-table::
   :header-rows: 1

   * - Safety benchmark
     - Metric
     - Score
   * - Overall safety average
     - ``overall_safety_average``
     - 66.7%
   * - WildGuardTest
     - ``inverted_micro_harm_lower``
     - 91.9%
   * - HarmBench
     - ``inverted_micro_asr_lower``
     - 57.5%
   * - ToxiGen (tiny)
     - ``safe_overall``
     - 90.2%
   * - XSTest
     - ``overall_accuracy``
     - 85.6%
   * - StrongReject (logprobs)
     - ``inverted_asr``
     - 78.8%
   * - BBQ
     - ``accuracy``
     - 66.5%
   * - WMDP
     - ``inverted_accuracy``
     - 47.5%

Next steps
----------

- To run Lizzy locally or on a GPU server, see :doc:`how-to-run-lizzy`.
- To plan hardware and memory for Lizzy variants, see
  :doc:`hardware-requirements`.
- To understand the training method, see :doc:`lizzy-training-and-evaluation`.
- To choose a quantized local-runtime model, see :doc:`lizzy-gguf`.
- For product details, see the `Lizzy model page <https://flower.ai/models/lizzy/>`_.
- For Flower Labs research, see `Flower Research <https://flower.ai/research/>`_.
- For enterprise deployments and custom work, see :doc:`enterprise`.

Lizzy 7B pages
--------------

.. toctree::
   :maxdepth: 1

   hardware-requirements
