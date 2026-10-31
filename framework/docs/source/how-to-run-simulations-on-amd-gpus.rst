:og:description: Run Flower simulations on AMD Instinct GPUs with ROCm. Configure Ray's AMD GPU discovery via ROCM_PATH and HIP_VISIBLE_DEVICES so the Simulation Runtime can place ClientApps.
.. meta::
    :description: Run Flower simulations on AMD Instinct GPUs with ROCm. Configure Ray's AMD GPU discovery via ROCM_PATH and HIP_VISIBLE_DEVICES so the Simulation Runtime can place ClientApps.

.. |clientapp_link| replace:: ``ClientApp``

.. _clientapp_link: ref-api/flwr.client.ClientApp.html

#############################
 Run simulations on AMD GPUs
#############################

Flower is framework-agnostic and performs no GPU compute of its own: all device work is
delegated to the ML framework your |clientapp_link|_ uses. On AMD Instinct GPUs,
acceleration therefore comes from a ROCm build of that framework (for example, PyTorch
for ROCm). Flower itself is pure Python, compiles no device code, and runs unmodified --
there is no separate AMD build of ``flwr``.

The Simulation Runtime is a second, independent consumer of the GPU. It uses Ray to
schedule ``ClientApp`` instances, and **Ray discovers AMD GPUs through a different
mechanism than your ML framework does**. The two can disagree: it is common for
``torch.cuda.is_available()`` to return ``True`` while Ray reports zero GPUs. When that
happens, every run that requests a GPU fraction fails before training starts. This guide
covers the AMD-specific setup that keeps the two in agreement.

.. note::

    PyTorch for ROCm exposes AMD GPUs through the ``cuda`` device namespace.
    ``torch.device("cuda")`` and ``torch.cuda.is_available()`` refer to the AMD GPU via
    HIP. No change to your ``ClientApp`` code is required.

The steps below were validated on a single-node AMD Instinct MI300X (gfx942) running
ROCm; other Instinct-class cards behave the same way. Consumer Radeon cards may
additionally need ``HSA_OVERRIDE_GFX_VERSION`` to be set for the ML framework -- a ROCm
concern rather than a Flower one; see AMD's ROCm documentation. Multi-GPU and multi-node
placement follow the same discovery rules described here, but are not covered
separately.

***********************************
 Check that ROCm can reach the GPU
***********************************

This guide assumes a working ROCm installation underneath. ROCm reaches the GPU through
``/dev/kfd`` and ``/dev/dri/renderD*``, which are normally owned by the ``render``
group:

.. code-block:: shell

    ls -l /dev/kfd   # e.g. crw-rw---- 1 root render ...
    id -nG           # must list the owning group

If your account is not in that group, nothing below will help: your ML framework and Ray
will *both* report zero GPUs. Add yourself to the ``render`` and ``video`` groups with
``sudo usermod -aG render,video "$USER"``, then start a new login session. If you run
Flower in a container, pass the devices through instead:

.. code-block:: shell

    docker run --device=/dev/kfd --device=/dev/dri \
        --group-add video --group-add render ...

``render`` matters as much as ``video``: ``/dev/kfd`` is owned by it, so a non-root
container that adds only ``video`` still sees zero GPUs. If your distribution owns the
devices with a differently named group, add that one instead -- the ``ls -l`` above
reports it.

The rest of this guide addresses the opposite, more confusing case: ROCm works and your
ML framework sees the GPU, but Ray does not.

******************************************
 Install the ROCm framework before Flower
******************************************

Install your ROCm-enabled ML framework first, following AMD's instructions for that
framework, and only then install Flower:

.. code-block:: shell

    # 1a. ROCm already installed system-wide (e.g. /opt/rocm): install a matching
    #     ROCm build of your framework. Replace "rocm6.4" with the index for your
    #     version; https://pytorch.org/get-started/locally/ reports the right one.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4

    # 1b. No system ROCm: install ROCm and the framework together from AMD's
    #     index. This produces the wheel-based layout discussed below. Replace
    #     gfx942 with your architecture, which `rocminfo | grep gfx` reports.
    pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
        "rocm[libraries,devel,device-gfx942]" "torch[device-gfx942]" \
        "torchvision[device-gfx942]"

    # 2. Flower with the simulation extra, which pulls in Ray
    pip install "flwr[simulation]"

.. important::

    Install the two in separate commands, in this order. Resolving them together can let
    ``pip`` select the default PyPI ``torch`` wheel, which is built for CUDA and will
    never see an AMD GPU. The installation succeeds and only fails later, at the point
    where your ``ClientApp`` tries to use the device.

.. important::

    Installing ROCm wheels into your own environment is not sufficient for the default
    ``flwr run`` workflow. By default, the SuperLink that ``flwr run`` starts installs
    each run's app dependencies into a fresh, isolated environment and activates it in
    place of your base environment. An app that declares a plain ``torch`` dependency
    can therefore resolve the default CUDA wheel from PyPI into that fresh environment
    even though both GPU checks below pass in your base environment. Choose one:

    - Pin the ROCm index in the app's own ``pyproject.toml``, so the install Flower
      performs for you resolves the same wheels:

      .. code-block:: toml

          [[tool.uv.index]]
          name = "rocm"
          url = "https://download.pytorch.org/whl/rocm6.4"
          explicit = true

          [tool.uv.sources]
          torch = { index = "rocm" }
          torchvision = { index = "rocm" }

    - Or preinstall every app dependency into your base environment and skip the per-run
      install entirely, by disabling it before the SuperLink starts:

      .. code-block:: shell

          export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1

      Set this *before* the first ``flwr run`` of the session -- a local SuperLink is
      started lazily on first use and, like ``ROCM_PATH`` and ``HIP_VISIBLE_DEVICES``
      below, keeps the environment it started with until it is stopped and restarted.
      The equivalent flag on a SuperLink you start yourself is
      ``--disable-runtime-dependency-installation``.

***********************************
 Point Ray at the ROCm SMI library
***********************************

Ray enumerates AMD GPUs by opening ``$ROCM_PATH/lib/librocm_smi64.so``, where
``ROCM_PATH`` defaults to ``/opt/rocm``. Two details matter:

- Ray opens the library by its **unversioned** name. A ROCm tree that ships only a
  versioned ``librocm_smi64.so.<N>`` (for example ``librocm_smi64.so.1``) and no
  ``librocm_smi64.so`` symlink is not sufficient.
- If you installed ROCm from Python wheels rather than system packages, there is usually
  no ``/opt/rocm`` at all, so ``ROCM_PATH`` must be set explicitly.

Confirm the library is where Ray will look for it:

.. code-block:: shell

    test -f "${ROCM_PATH:-/opt/rocm}/lib/librocm_smi64.so" && echo "ROCm SMI found"

If that prints nothing, set ``ROCM_PATH`` to a ROCm root that contains the unversioned
symlink and re-check. When ROCm is installed from wheels, the ``rocm-sdk`` tool reports
the correct root:

.. code-block:: shell

    export ROCM_PATH="$(rocm-sdk path --root)"

``rocm-sdk`` is provided by the ``devel`` extra. If it reports that ``rocm[devel]`` is
not installed, install it -- that package supplies the root you need here.

.. important::

    A wheel-based ROCm installation may expose more than one root. Only the root that
    ships the unversioned ``librocm_smi64.so`` symlink works for Ray: in current AMD
    wheel layouts that is ``_rocm_sdk_devel``, while ``_rocm_sdk_core`` ships only the
    versioned ``librocm_smi64.so.<N>``. Your ML framework locates HIP independently and
    works with either, which is why this misconfiguration shows up as a Ray-only
    failure.

    Published ROCm container recipes commonly set ``ROCM_PATH`` to the *core* root,
    which is correct for the ML framework but leaves Ray unable to enumerate GPUs. If
    you adapt such an image for the Simulation Runtime, override ``ROCM_PATH`` for Ray.

*************************************
 Set ``HIP_VISIBLE_DEVICES`` for Ray
*************************************

Many GPU hosts preset ``ROCR_VISIBLE_DEVICES`` in the login environment. Ray rejects
that variable: when ``ROCR_VISIBLE_DEVICES`` is set and ``HIP_VISIBLE_DEVICES`` is not,
``import ray`` raises ``RuntimeError: Please use HIP_VISIBLE_DEVICES instead of
ROCR_VISIBLE_DEVICES``.

The failure occurs at **import** time, not at ``ray.init()``. Because every Ray worker
re-imports Ray on startup, this also surfaces indirectly as workers that never register.

Satisfy Ray without disturbing ``ROCR_VISIBLE_DEVICES``, by deriving
``HIP_VISIBLE_DEVICES`` from how many devices it already lists rather than copying its
values. Only do this when ``HIP_VISIBLE_DEVICES`` is not already set -- the import error
this works around only occurs in that case, and an existing value may deliberately
select a subset of the ROCr-visible devices. Test for ``ROCR_VISIBLE_DEVICES`` being
*set*, not merely non-empty: a scheduler can export it as an empty string to hide every
device, and that case must still satisfy Ray's import guard by propagating an equally
empty ``HIP_VISIBLE_DEVICES`` -- counting entries in an empty string would otherwise
synthesize ``HIP_VISIBLE_DEVICES=0`` and wrongly reveal a device. The same applies to
the nonempty ``-1`` sentinel some schedulers use for "no device": it contains no comma,
so the comma-counting branch would otherwise read it as one listed GPU and export
``HIP_VISIBLE_DEVICES=0``, letting Ray advertise and schedule onto a device ROCr never
actually exposed:

.. code-block:: shell

    if [ -n "${ROCR_VISIBLE_DEVICES+x}" ] && [ -z "${HIP_VISIBLE_DEVICES+x}" ]; then
        if [ -z "$ROCR_VISIBLE_DEVICES" ] || [ "$ROCR_VISIBLE_DEVICES" = "-1" ]; then
            export HIP_VISIBLE_DEVICES=""
        else
            _n=$(($(tr -cd ',' <<<"$ROCR_VISIBLE_DEVICES" | wc -c) + 1))
            export HIP_VISIBLE_DEVICES="$(seq -s, 0 $((_n - 1)))"
            unset _n
        fi
    fi

.. important::

    Do not copy ``ROCR_VISIBLE_DEVICES``'s values into ``HIP_VISIBLE_DEVICES``, and do
    not simply unset ``ROCR_VISIBLE_DEVICES`` -- on a shared host it may be how a
    scheduler isolates your job's GPUs from everyone else's, so removing it can expose
    devices that are not yours. The two masks instead *compose* in a specific order:
    ROCr applies its filter first and renumbers whatever remains starting from ``0``,
    and only then does ``HIP_VISIBLE_DEVICES`` index into that renumbered set -- not
    into the original physical IDs. With ``ROCR_VISIBLE_DEVICES=1``, the sole visible
    GPU becomes ordinal ``0``, so setting ``HIP_VISIBLE_DEVICES=1`` to match asks for a
    device that no longer exists and Ray sees no GPU at all.

    The snippet above sidesteps this: whatever ``ROCR_VISIBLE_DEVICES`` already
    restricts you to, after filtering there are exactly that many devices, numbered
    contiguously from ``0``. Selecting all of them (``0`` through ``n-1`` in
    ``HIP_VISIBLE_DEVICES``) reproduces the original restriction exactly, while leaving
    ``ROCR_VISIBLE_DEVICES`` itself untouched. The snippet only synthesizes this value
    when ``HIP_VISIBLE_DEVICES`` is unset: an existing value is already expressed in the
    renumbered space and may deliberately select a subset of it -- for example, to
    dedicate only one of two ROCr-visible devices to this process -- so it is left
    alone.

.. important::

    These exports apply only to the shell you run them in, and the local SuperLink that
    ``flwr run`` manages copies its environment once, at startup. If a SuperLink from an
    earlier, uncorrected session is still running, ``flwr run`` reuses it rather than
    starting a fresh one -- which creates a trap: the two verification commands below
    run in *your* corrected shell and pass, while every subsequent ``flwr run`` still
    talks to a SuperLink holding the old, broken environment. A passing check is
    therefore not proof that a run will succeed.

    Stop the stale SuperLink before trusting those checks or applying simulation
    configuration, using the inspect-then-terminate procedure in
    :ref:`stop-background-local-superlink`. Matching on the bare process name risks
    killing every ``flower-superlink`` process the account owns, including unrelated
    runs; that procedure scopes the match to the managed port instead.

    ``flwr run`` starts a new one automatically, which copies the corrected environment
    of the shell that invokes it. There is no dedicated ``flwr`` command to stop just
    the SuperLink process; ``flwr stop <run-id>`` stops a run, not the SuperLink itself.

    If you manage a SuperExec separately from the one SuperLink auto-launches -- for
    example, in a deployment that is not this local, single-command setup -- restarting
    SuperLink alone does not refresh it. Restart that SuperExec process too.

    The same singleton reuse means a ``HIP_VISIBLE_DEVICES`` export does not isolate one
    run from another. ``flwr run`` is a client of the already-running SuperLink, not a
    fresh launch of it: every run that reuses that SuperLink shares the one mask it
    started with, so two simulations started from shells with different exports still
    submit to the same process and can target the same physical GPU. A mask is a
    property of the SuperLink, not of an individual run -- it only changes when the
    SuperLink itself is stopped and restarted, same as above. Run simulations that need
    different masks sequentially, stopping the SuperLink between them, rather than
    concurrently.

*************************************
 Verify that Ray detects the AMD GPU
*************************************

Checking your ML framework is not enough, since it and Ray discover devices
independently. Check both.

First, the framework:

.. code-block:: shell

    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

Then Ray, which is what the Simulation Runtime actually schedules against:

.. code-block:: shell

    python -c "import ray; ray.init(); print(ray.cluster_resources()); ray.shutdown()"

The output must contain a non-zero ``GPU`` entry:

.. code-block:: text

    {'CPU': 8.0, 'GPU': 1.0, 'accelerator_type:AMD-Instinct-MI300X-OAM': 1.0, ...}

The ``GPU`` entry is the one that matters. Ray derives ``accelerator_type`` from a
hardcoded table of device IDs, so a card missing from that table is reported without an
``accelerator_type`` label -- it is still counted and still scheduled normally.

Only when ``GPU`` is present and non-zero will a run configured with
``client-resources-num-gpus`` greater than zero be able to place a ``ClientApp``. With
both checks passing, configure and run your simulation exactly as on any other
accelerator -- see :doc:`how-to-run-simulations`:

.. code-block:: shell

    flwr federation simulation-config --num-supernodes 8 --client-resources-num-gpus 0.25
    flwr run . --stream

*****************
 Troubleshooting
*****************

``ActorPool is empty. Stopping Simulation.``
============================================

Ray detected no GPU while your ``ClientApp`` requested a fraction of one. Despite the
message, the process does not always exit: the run can appear to hang after the error is
printed. If that happens, interrupt it and scroll back -- the warning below is emitted
before the hang and is the part that names the cause.

The preceding warning names the mismatch:

.. code-block:: text

    WARNING : Ray reports no GPU resources (GPUs=0) but each client requests
    num_gpus=0.25, which no node can satisfy. This does not by itself mean GPU
    discovery failed: the cluster may have no GPU, or Ray's GPU resources may
    have been limited deliberately (for example `--init-args-num-gpus 0`). Set
    `num_gpus=0` if no GPU is expected, otherwise review that limit before
    suspecting discovery.

Three different situations produce this warning, and only one of them calls for the
ROCm fixes elsewhere in this guide:

- The cluster genuinely has no GPU. Set ``client-resources-num-gpus`` to ``0`` and move
  on.
- Ray's GPU resources were limited deliberately, for example via
  ``flwr federation simulation-config --init-args-num-gpus 0``. Lowering
  ``client-resources-num-gpus`` will not help here either -- no non-zero value can be
  satisfied by a cluster reporting ``GPUs=0`` -- but neither will ``ROCM_PATH`` or
  ``HIP_VISIBLE_DEVICES``, since the limit is applied at ``ray.init()`` and never
  reaches ROCm. Review, or raise, the configured ``init-args-num-gpus`` value instead.
- ROCm reaches the GPU but Ray did not discover it. Work through `Point Ray at the ROCm
  SMI library`_ and `Set HIP_VISIBLE_DEVICES for Ray`_, then re-run the verification
  above.

Compare what Ray actually detects (``ray.cluster_resources()``, as verified above)
against the ``--init-args-num-gpus`` value you configured to tell the second case apart
from the third.

When ``/dev/kfd`` is present, Flower follows that warning with a second one naming the
``ROCM_PATH`` requirement directly.

Note that ``torch.cuda.is_available()`` commonly reports ``True`` in exactly this
situation, because your ML framework locates HIP without consulting ``ROCM_PATH``.

Both the ML framework and Ray report zero GPUs
==============================================

When ``torch.cuda.is_available()`` is ``False`` *and* Ray reports no GPU, the problem is
below Flower: ROCm itself cannot reach the device. The usual cause is that your account
lacks access to ``/dev/kfd``, or that a container was started without the device flags.
See `Check that ROCm can reach the GPU`_. The ``ROCM_PATH`` and ``HIP_VISIBLE_DEVICES``
steps in this guide apply only once ROCm works for your ML framework.

``Please use HIP_VISIBLE_DEVICES instead of ROCR_VISIBLE_DEVICES``
==================================================================

See `Set HIP_VISIBLE_DEVICES for Ray`_. Note that this is raised when Ray is imported,
so it can appear either directly in your own process or indirectly as Ray workers that
fail to register.

A run completes but the GPU stays idle
======================================

Restarting the local SuperLink resets the simulation configuration to its defaults,
including ``client-resources-num-gpus`` back to ``0.0``, and the run then completes on
CPU. Apply ``flwr federation simulation-config`` *after* any restart, and confirm the
active value before concluding that a run exercised the GPU:

.. code-block:: shell

    flwr federation ls --federation @none/default
