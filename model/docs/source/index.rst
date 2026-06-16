Flower Model Documentation
==========================

Welcome to Flower Model documentation. `Flower <https://flower.ai>`_ builds
tools and models for collaborative AI.

Join the Flower Community
-------------------------

The Flower Community includes researchers, engineers, students, professionals,
academics, and other people building collaborative AI systems.

.. button-link:: https://flower.ai/join-slack
    :color: primary
    :shadow:

    Join us on Slack


Flower Models
-------------

Flower models are open-weight releases from Flower Labs. These pages describe
model capabilities, release formats, deployment options, evaluation results,
safety information, and limitations.

With Flower Model documentation, you can:

- Learn the intended uses and limitations of Flower Labs model releases.
- Choose between full-precision and quantized model artifacts.
- Run models locally, through Python libraries, or with serving runtimes.
- Review architecture, evaluation, and safety information.

The first documented model is Lizzy 7B, a UK-oriented assistant model available
as both a BF16 Safetensors checkpoint and GGUF quantizations.

For product details, see the `Lizzy model page <https://flower.ai/models/lizzy/>`_.
For background on Flower Labs research, see `Flower Research <https://flower.ai/research/>`_.


Model guides
~~~~~~~~~~~~

Model guides provide practical information about Flower Labs models.

.. toctree::
  :maxdepth: 1
  :caption: Models

  lizzy-7b
  enterprise


How-to guides
~~~~~~~~~~~~~

How-to guides provide step-by-step instructions to help you accomplish specific tasks.

.. toctree::
  :maxdepth: 1
  :caption: How-to Guides

  how-to-run-lizzy


Explanations
~~~~~~~~~~~~

Explanations provide background on model evaluation, release formats, and
deployment trade-offs.

.. toctree::
  :maxdepth: 1
  :caption: Explanations

  lizzy-training-and-evaluation
  lizzy-gguf
  troubleshooting
