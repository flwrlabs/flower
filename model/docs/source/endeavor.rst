Endeavor 1.0
============

Endeavor 1.0 is Flower Labs' frontier-class language model. It is designed
for advanced reasoning, coding assistance, and long-horizon agent work.
It is a core generalist model across products, agents, and
complex workflows rather than a model narrowly optimized for a single
benchmark family. Endeavor 1.0 is the second model in the Flower model program,
following Lizzy 7B, the sovereign 7B model built for UK use.

Endeavor 1.0 is a production-ready preview release, initially available to a
select group of organizations and partners as we expand access. Here,
``production-ready`` describes the supported managed and private deployment
paths; ``preview`` describes the release's limited, request-based availability.

Endeavor 1.0 is available in two operating modes:

- A Flower-managed service: call Endeavor 1.0 from existing applications and agent
  frameworks through familiar model APIs and response formats.
- Private deployment: self-host Endeavor 1.0 in your own environment, with Flower
  support across your data, infrastructure, and systems.

Private deployment is a supported operating arrangement and
Endeavor 1.0 is not currently provided as a public open-weight release.

At a glance
-----------

.. list-table::
   :header-rows: 1

   * - Property
     - Value
   * - Publisher
     - Flower Labs
   * - Family
     - Endeavor
   * - Version
     - 1.0
   * - Positioning
     - Frontier-class generalist model for reasoning, coding, and
       long-horizon agent work
   * - Access
     - Flower-managed API, or private deployment on infrastructure you
       control; by request during the 1.0 preview
   * - Model artifacts
     - Availability and formats are confirmed during private-deployment
       onboarding; no public checkpoint is listed for download during the
       1.0 preview
   * - Predecessor
     - Lizzy 7B, a sovereign 7B model built for UK use

Getting access
--------------

During the Endeavor 1.0 preview, access is available by request as we
onboard a limited number of organizations and partners. To request access,
use the `Endeavor 1.0 access form <https://form.typeform.com/to/KmUCC8xu>`_.

Preview interface and support
-----------------------------

Before integrating, approved participants should confirm deployment-specific
details during onboarding: the endpoint and credentials, model identifier,
supported API and response formats, context limits, tool and streaming support,
usage limits, and support channels. These details can vary between the
Flower-managed service and a private deployment.

The complete technical contract is provided only to approved participants
during onboarding. It is not published as part of the public Endeavor 1.0
documentation.

Architecture and configuration
------------------------------

Endeavor 1.0 is optimized as a complete AI system rather than as a set of weights
alone. It is tuned for how the model reasons at inference time, how it
builds and maintains context, how it uses tools, and how it checks,
revises, and recovers when a step fails.

These behaviors are developed and evaluated across multiple custom coding
and agent harnesses, varying tool sets, interaction formats, and execution
environments, so that performance transfers across real workflows rather
than depending on a single benchmark setup.

In the managed service, Flower operates deployment, scaling, and model
operations. In a private deployment, you run Endeavor 1.0 in your own environment
and choose when to adopt upgrades, retaining the option to operate Endeavor 1.0
independently of any single provider.

Training approach
-----------------

Endeavor 1.0 is built from a different starting point: it builds on mature,
widely available capabilities from leading open-weight models, and adds
the capabilities developed through the Flower model program on top of
that foundation.

- Mature foundation
  General language understanding, public knowledge, and common coding
  patterns from leading open-weight models, so Endeavor 1.0 builds on what the
  wider ecosystem already does well rather than recreating it from the
  ground up.
- Flower-specific capabilities
  UK-specialist knowledge and reasoning carried over from Lizzy, together
  with new model behaviors, specialist capabilities, and training advances
  developed for Endeavor 1.0.
- Continual pre-training, targeted post-training, and model integration
  These stages bring the two sets of strengths together and allow Endeavor 1.0 to
  be refined over time.
- Signals from real enterprise work
  FlowerBench enables repeatable evaluation on real, high-value enterprise
  workflows without moving the underlying proprietary data. Tasks are
  contributed by organizations in the opt-in Flower Enterprise Evaluation
  Network and run inside their own environments. The resulting end-to-end
  signals guide Endeavor 1.0's evaluation design, harness development,
  post-training, and system-level improvements.

Evaluation highlights
---------------------

Endeavor 1.0 is evaluated directly against leading closed models from OpenAI and
Anthropic, as well as the strongest open-weight alternatives including
Kimi K3 and Nemotron 3 Ultra. The table below shows four core evaluations.

.. list-table::
   :header-rows: 1

   * - Benchmark
     - Endeavor 1.0
     - GPT-5.6 Sol
     - Claude Fable 5
     - Kimi K3
     - Nemotron 3 Ultra
   * - GPQA
     - 92.0
     - **94.1**
     - 92.6
     - 93.5
     - 86.7
   * - HumanEval
     - **98.2**
     - 95.1
     - 97.0
     - 96.3
     - 96.3
   * - IFEval
     - 94.1
     - **95.9**
     - 91.7
     - 92.8
     - 91.9
   * - AIME 2026
     - **99.9**
     - **99.9**
     - **99.9**
     - 96.7
     - 94.2

Endeavor 1.0 records the highest HumanEval score in the comparison set, matches
GPT-5.6 Sol and Claude Fable 5 on AIME 2026, leads Kimi K3 on three of the
four benchmarks (HumanEval, IFEval, AIME 2026), and leads Nemotron 3 Ultra
on all four. On GPQA, GPT-5.6 Sol, Claude Fable 5, and Kimi K3 score above
Endeavor 1.0; on IFEval, GPT-5.6 Sol scores above Endeavor 1.0.

Scores for the comparison models are as reported by the respective vendors
and third-party evaluations (Artificial Analysis, Moonshot AI, NVIDIA, and
Thinking Machines Lab) under different harnesses and settings; treat the
cross-model comparison as indicative rather than a controlled head-to-head
measurement.

The Endeavor 1.0 scores reflect Flower's internal evaluation of the launch
configuration available on 31 August 2026. Representative public sources used
for the comparison set include the `Kimi K3 report
<https://github.com/MoonshotAI/Kimi-K3>`_, the `NVIDIA Nemotron 3 Ultra report
<https://arxiv.org/html/2606.15007v1>`_, and the `Thinking Machines Lab Inkling
evaluation <https://thinkingmachines.ai/news/introducing-inkling/>`_. These
sources do not use one common evaluation harness.

Additional agentic and reasoning evaluations will be available soon.

Safety and limitations
----------------------

Endeavor 1.0 can make mistakes. It can produce incorrect, outdated,
incomplete, or over-confident responses. Verify important outputs before using
them, and do not rely on Endeavor 1.0 as the sole basis for medical, legal,
financial, safety-critical, or other high-impact decisions. Higher-risk
workflows require appropriate human oversight, domain review, access controls,
monitoring, and downstream moderation.

Endeavor 1.0 is a broad generalist model, but its tone, assumptions, and
responses can still reflect limitations or biases in its training data and
system design.

Use of the Flower-managed service is subject to the `Flower Acceptable Use
Policy <https://flower.ai/pdfs/legal/flower_labs_gmbh_terms_of_service_aup_052025.pdf>`_.
Deployment-specific safety, monitoring, and support requirements will be detailed
during onboarding.

Next steps
----------

- To see the first model in the Flower model program, see
  :doc:`lizzy-7b`.
- For enterprise deployments and custom work, see :doc:`enterprise`.
- For product details and the datasheet, see the `Endeavor 1.0 model page
  <https://flower.ai/models/endeavor/>`_.
- For Flower Labs research, see `Flower Research
  <https://flower.ai/research/>`_.
