Lizzy Training and Evaluation
=============================

Lizzy 7B was produced through a multi-stage training and evaluation process.
This page gives a high-level summary of the method and reported evaluation
areas without depending on non-public implementation details.

Training method
---------------

The release uses four broad stages:

Pre-training
  Large-scale public text, document, code, math, and encyclopedic corpora.

Supervised fine-tuning
  Instruction-following, dialogue, reasoning, and tool-use examples.

Direct preference optimisation
  Preference pairs used to improve helpfulness, style, and answer quality.

Reinforcement learning with verifiable rewards
  Targeted behavioural refinement with verifiable reward signals.

Data mix
--------

The release describes a mix of:

- broad public text and knowledge sources
- instruction and preference data
- UK-specific examples and preference signals

The full training data mix is not redistributed with the public checkpoints.

Evaluation
----------

The release reports UK-oriented benchmarks, general knowledge and reasoning
benchmarks, coding benchmarks, math benchmarks, instruction-following
evaluations, and safety evaluations.

The reported comparison set includes EuroLLM 9B and Apertus 8B. Lizzy 7B leads
that local baseline set on most represented reasoning, math, knowledge, and
coding rows, while trailing on the Britishness MCQ recall-style benchmark.

Safety evaluation
-----------------

The release reports a task-level safety summary across WildGuardTest,
HarmBench, ToxiGen, XSTest, StrongReject, BBQ, and WMDP. These numbers are
evaluation signals, not guarantees. Production deployments should include
human oversight where appropriate, policy checks, monitoring, and downstream
moderation.

Release and serving workflow
----------------------------

The GGUF release provides llama.cpp-compatible files for local deployment.
The BF16 Safetensors release is the target for full-precision GPU serving,
fine-tuning, and Transformers or vLLM workflows.
