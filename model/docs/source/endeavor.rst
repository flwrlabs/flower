.. meta::
    :description: Explore Flower Endeavor, a language model for reasoning, coding, and agent workflows, with managed API access and private deployment.
    :property=og:description: Explore Flower Endeavor, a language model for reasoning, coding, and agent workflows, with managed API access and private deployment.

##########
 Endeavor
##########

Endeavor is Flower Labs' frontier-class language model. It is designed for advanced
reasoning, coding assistance, and long-horizon agent work. It is a core generalist model
across products, agents, and complex workflows. Endeavor is the second model in the
Flower model program, following Lizzy 7B, the sovereign 7B model built for UK use.

.. _endeavor-versions:

********************
 Available versions
********************

.. list-table::
    :header-rows: 1

    - - Version
      - Status
    - - Endeavor |endeavor_version|
      - Latest release, available by request
    - - Endeavor 1.0
      - Available by request

The client guides below use Endeavor |endeavor_version|.

Endeavor is available in two operating modes:

- A Flower-managed service: call Endeavor from existing applications and agent
  frameworks through familiar model APIs and response formats.
- Private deployment: self-host Endeavor in your own environment, with Flower support
  across your data, infrastructure, and systems.

*************
 At a glance
*************

.. list-table::
    :header-rows: 1

    - - Property
      - Value
    - - Publisher
      - Flower Labs
    - - Family
      - Endeavor
    - - Versions
      - |endeavor_version| (latest) and 1.0
    - - Positioning
      - Frontier-class generalist model for reasoning, coding, and long-horizon agent
        work
    - - Access
      - Flower-managed API, or private deployment on infrastructure you control,
        available by request
    - - API base URL (Flower-managed service)
      - ``https://api.flower.ai/v1``
    - - Model name (latest version)
      - :substitution-code:`|endeavor_model|`
    - - Context window
      - 128,000 tokens in the supplied Codex configuration, with automatic compaction at
        96,000 tokens; the supported API limit varies by deployment. See :ref:`Context
        window and limits <endeavor-context-limits>`.
    - - Maximum context length
      - 1M tokens
    - - Model artifacts
      - Private distribution, with availability and formats confirmed during onboarding
    - - Predecessor
      - Lizzy 7B, a sovereign 7B model built for UK use

****************
 Getting access
****************

Endeavor access is available by request as we onboard organizations and partners. To
request access, use the `Endeavor access form
<https://flowerlabs.typeform.com/to/jlniHsuy>`_.

Once you have a Flower API key, connect Endeavor to :doc:`ChatGPT/Codex
<endeavor-chatgpt-codex>` or :doc:`OpenCode <endeavor-opencode>`. Both guides cover CLI
and desktop use with the Flower-managed service.

***********************
 Interface and support
***********************

Before integrating, approved participants should confirm deployment-specific details
during onboarding: the endpoint and credentials, model identifier, supported API and
response formats, context limits, tool and streaming support, usage limits, and support
channels. These details can vary between the Flower-managed service and a private
deployment.

Approved participants receive the complete technical contract during onboarding.

***************************
 Context window and limits
***************************

.. _endeavor-context-limits:

The supplied Codex configuration uses a 128,000-token context window, with automatic
compaction configured at 96,000 tokens. These settings control the client's context
management. Effective API request limits depend on the deployment, upstream provider
route, requested output budget, and internal processing overhead. Flower Endeavor
supports up to a 1M-token context window. Confirm the supported limit for your
deployment during onboarding.

********************************
 Architecture and configuration
********************************

Endeavor is optimized as a complete AI system. It is tuned for how the model reasons at
inference time, how it builds and maintains context, how it uses tools, and how it
checks, revises, and recovers when a step fails.

These behaviors are developed and evaluated across multiple custom coding and agent
harnesses, varying tool sets, interaction formats, and execution environments, so that
performance transfers across real workflows.

In the managed service, Flower operates deployment, scaling, and model operations. In a
private deployment, you run Endeavor in your own environment and choose when to adopt
upgrades, retaining the option to operate Endeavor independently of any single provider.

*******************
 Training approach
*******************

Endeavor is built from a different starting point: it builds on mature, widely available
capabilities from leading open-weight models, and adds the capabilities developed
through the Flower model program on top of that foundation.

- **Mature foundation**. General language understanding, public knowledge, and common
  coding patterns from leading open-weight models, so Endeavor builds on what the wider
  ecosystem already does well.
- **Flower-specific capabilities**. UK-specialist knowledge and reasoning carried over
  from Lizzy, together with new model behaviors, specialist capabilities, and training
  advances developed for Endeavor.
- **Continual pre-training, targeted post-training, and model integration**. These
  stages bring the two sets of strengths together and allow Endeavor to be refined over
  time.
- **Signals from real enterprise work**. FlowerBench enables repeatable evaluation on
  real, high-value enterprise workflows without moving the underlying proprietary data.
  Tasks are contributed by organizations in the opt-in Flower Enterprise Evaluation
  Network and run inside their own environments. The resulting end-to-end signals guide
  Endeavor's evaluation design, harness development, post-training, and system-level
  improvements.

***********************
 Evaluation highlights
***********************

The table below places Endeavor 1.1's and Endeavor 1.0's results on five core
evaluations alongside separately reported scores for leading closed and open-weight
models, including GPT-5.6 Sol, Claude Fable 5, Kimi K3, and Nemotron 3 Ultra.

.. list-table::
    :header-rows: 1

    - - Benchmark
      - Endeavor 1.1
      - Endeavor 1.0
      - GPT-5.6 Sol
      - Claude Fable 5
      - Kimi K3
      - Nemotron 3 Ultra
    - - GPQA
      - **94.4**
      - 92.0
      - 94.1
      - 92.6
      - 93.5
      - 86.7
    - - HumanEval
      - **99.9**
      - 98.2
      - 95.1
      - 97.0
      - 96.3
      - 96.3
    - - IFEval
      - 94.5
      - 94.1
      - **95.9**
      - 91.7
      - 92.8
      - 91.9
    - - AIME 2026
      - **99.9**
      - **99.9**
      - **99.9**
      - **99.9**
      - 96.7
      - 94.2
    - - MMMU-Pro
      - 81.12
      - n/a
      - **83.0**
      - 81.2
      - 81.6
      - n/a

Within this cross-source comparison, Endeavor 1.1's reported GPQA and HumanEval scores
are the highest. Endeavor 1.0's reported AIME 2026 score equals those reported for
GPT-5.6 Sol and Claude Fable 5; its scores exceed the reported Kimi K3 scores on three
of four text benchmarks (HumanEval, IFEval, AIME 2026) and the reported Nemotron 3 Ultra
scores on all four. The reported GPQA scores for GPT-5.6 Sol, Claude Fable 5, and Kimi
K3 exceed Endeavor 1.0's; the reported IFEval score for GPT-5.6 Sol also exceeds
Endeavor 1.0's.

Scores for the comparison models are as reported by the respective vendors and
third-party evaluations (Artificial Analysis, Moonshot AI, NVIDIA, and Thinking Machines
Lab) under different harnesses and settings. Treat the cross-model comparison as
indicative.

The Endeavor 1.0 scores reflect Flower's internal evaluation of the launch configuration
available on 31 August 2026. Representative public sources used for the comparison set
include the `Kimi K3 report <https://github.com/MoonshotAI/Kimi-K3>`_, the `NVIDIA
Nemotron 3 Ultra report <https://arxiv.org/html/2606.15007v1>`_, and the `Thinking
Machines Lab Inkling evaluation
<https://thinkingmachines.ai/news/introducing-inkling/>`_.

Additional agentic and reasoning evaluations will be available soon.

************************
 Safety and limitations
************************

Endeavor can make mistakes. It can produce incorrect, outdated, incomplete, or
over-confident responses. Verify important outputs before using them, and do not rely on
Endeavor as the sole basis for medical, legal, financial, safety-critical, or other
high-impact decisions. Higher-risk workflows require appropriate human oversight, domain
review, access controls, monitoring, and downstream moderation.

Endeavor is a broad generalist model, but its tone, assumptions, and responses can still
reflect limitations or biases in its training data and system design.

Use of the Flower-managed service is subject to the `Flower Acceptable Use Policy
<https://flower.ai/pdfs/legal/flower_labs_gmbh_terms_of_service_aup_052025.pdf>`_.
Deployment-specific safety, monitoring, and support requirements will be detailed during
onboarding.

***********************
 Guides and Next Steps
***********************

- For product details and the datasheet, see the `Endeavor model page
  <https://flower.ai/models/endeavor/>`_.
- For enterprise deployments and custom work, see :doc:`enterprise`.
- To see the first model in the Flower model program, see :doc:`lizzy-7b`.
- For Flower Labs research, see `Flower Research <https://flower.ai/research/>`_.

*****************
 Endeavor guides
*****************

.. toctree::
    :maxdepth: 1

    endeavor-chatgpt-codex
    endeavor-opencode
    endeavor-api-key
