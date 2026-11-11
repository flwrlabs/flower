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

Confirm the library is where Ray will look for it. Use the same unset-vs-empty test as
``ROCR_VISIBLE_DEVICES`` below rather than a plain ``${ROCM_PATH:-/opt/rocm}``
substitution: ``:-`` falls back to ``/opt/rocm`` both when ``ROCM_PATH`` is unset *and*
when it is set to an empty string, but Ray 2.55.1 only applies its own ``/opt/rocm``
default when ``ROCM_PATH`` is completely absent from the environment -- an explicit
``ROCM_PATH=""`` is kept as-is and Ray ends up opening ``/lib/librocm_smi64.so``
instead. A check using ``:-`` would still find the library under ``/opt/rocm`` and
misleadingly print "ROCm SMI found" in that case, even though Ray reports zero GPUs:

.. code-block:: shell

    if [ -n "${ROCM_PATH+x}" ]; then
        _rocm_path="$ROCM_PATH"
    else
        _rocm_path="/opt/rocm"
    fi
    test -f "$_rocm_path/lib/librocm_smi64.so" && echo "ROCm SMI found"
    unset _rocm_path

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
values. Only *synthesize* a value when ``HIP_VISIBLE_DEVICES`` is not already set -- the
import error this works around only occurs in that case, and an existing value may
deliberately select a subset of the ROCr-visible devices. Test for
``ROCR_VISIBLE_DEVICES`` being *set*, not merely non-empty: a scheduler can export it as
an empty string to hide every device, and that case must still satisfy Ray's import
guard by propagating an equally empty ``HIP_VISIBLE_DEVICES`` -- counting entries in an
empty string would otherwise synthesize ``HIP_VISIBLE_DEVICES=0`` and wrongly reveal a
device. The same applies to the nonempty ``-1`` sentinel some schedulers use for "no
device", and to Slurm's own ``NoDevFiles`` sentinel -- documented for its GPU GRES
integration -- which it exports instead when an allocation has no bound GPU device
files: both contain no comma, so the comma-counting branch would otherwise read either
one as one listed GPU and export ``HIP_VISIBLE_DEVICES=0``, letting Ray advertise and
schedule onto a device ROCr never actually exposed.

When ``HIP_VISIBLE_DEVICES`` is already set, do not simply trust it, either: validate
that every value it names is actually an ordinal ROCr's filtering leaves available,
rather than assuming any existing nonempty value is already expressed correctly. A
scheduler, or an earlier, less careful shell setup, can copy a *physical* device ID into
both variables -- for example ``ROCR_VISIBLE_DEVICES=3`` alongside
``HIP_VISIBLE_DEVICES=3``. Left alone, that pair passes every check below while still
being broken: ROCr's filter exposes that single device as ordinal ``0``, so
``HIP_VISIBLE_DEVICES=3`` asks for an ordinal that no longer exists, and the
verification steps later in this guide still report zero GPUs with no indication of why.

A value can also be in range and syntactically well-formed while still being wrong: a
repeated ordinal, such as ``HIP_VISIBLE_DEVICES=0,0``, passes a check that only looks at
each value in isolation. Ray 2.55.1 sizes its advertised GPU resources from the length
of this mask, so a duplicate ordinal creates two schedulable slots that both map to the
same ROCr-visible device, letting concurrent ClientApps collide on it instead of each
getting a distinct GPU. A duplicate can also hide behind a leading-zero alias -- ``0``
and ``00`` are the same ordinal to HIP, but two distinct strings to a naive comparison
-- so the snippet normalizes each value before comparing, rather than comparing the
spellings it was given. It also rejects the mask in that case rather than accepting a
syntactically-valid-looking duplicate.

A trailing comma is a similar trap: Bash's ``read -ra`` silently drops a single empty
field left by a trailing separator, so ``HIP_VISIBLE_DEVICES=0,1,`` parses to the same
two values as ``0,1`` and would otherwise pass every check above unchanged. Ray does not
make that same allowance -- it splits the raw string itself and sizes its GPU resources
from the resulting field count, including the empty one -- so it can advertise a slot
that no ordinal maps to. The snippet checks the raw string for a trailing comma before
trusting what ``read -ra`` parsed from it, and rejects the mask if one is present.

This snippet requires **Bash**: it uses here-strings (``<<<``), ``read -ra``, and ``[[
... =~ ... ]]`` regex matching, none of which are POSIX ``sh`` features. Run it with
``bash``, not with ``sh`` or a POSIX-only shell like ``dash`` -- on Debian and Ubuntu in
particular, ``/bin/sh`` is ``dash``, so pasting this directly into a ``#!/bin/sh``
script or a ``sh -c "..."`` invocation fails before ``HIP_VISIBLE_DEVICES`` is ever
exported, silently leaving Ray's original import failure in place:

.. code-block:: bash

    if [ -n "${ROCR_VISIBLE_DEVICES+x}" ]; then
        if [ -z "$ROCR_VISIBLE_DEVICES" ] || [ "$ROCR_VISIBLE_DEVICES" = "-1" ] ||
            [ "$ROCR_VISIBLE_DEVICES" = "NoDevFiles" ]; then
            _n=0
        else
            _n=$(($(tr -cd ',' <<<"$ROCR_VISIBLE_DEVICES" | wc -c) + 1))
        fi
        if [ -z "${HIP_VISIBLE_DEVICES+x}" ]; then
            if [ "$_n" -gt 0 ]; then
                export HIP_VISIBLE_DEVICES="$(seq -s, 0 $((_n - 1)))"
            else
                export HIP_VISIBLE_DEVICES=""
            fi
        else
            _bad=0
            if [[ "$HIP_VISIBLE_DEVICES" == *, ]]; then
                # A trailing comma leaves `read -ra` one field short of what Ray
                # gets from splitting the raw string itself, so check the raw
                # string before trusting the parsed array's field count.
                _bad=1
            fi
            _norm_ids=()
            IFS=',' read -ra _hip_ids <<<"$HIP_VISIBLE_DEVICES"
            for _id in "${_hip_ids[@]}"; do
                if ! [[ "$_id" =~ ^[0-9]+$ ]]; then
                    _bad=1
                    continue
                fi
                # Compare digit-string lengths rather than `[ "$_id" -ge "$_n"
                # ]` directly: an all-digit ordinal outside Bash's integer
                # range (e.g. a mistyped 30-digit value) makes that
                # comparison fail with "integer expression expected" and a
                # nonzero exit status -- which the original `||` chain
                # treated as false, silently accepting an out-of-range
                # ordinal. `_n` is always small, so a trimmed ordinal with
                # more digits than `_n` is unambiguously too large without
                # needing arithmetic on it, and one with the same digit
                # count is bounded by `_n`'s own magnitude, so comparing it
                # numerically is safe.
                _trimmed_id="${_id#"${_id%%[!0]*}"}"
                [ -z "$_trimmed_id" ] && _trimmed_id=0
                if [ "${#_trimmed_id}" -gt "${#_n}" ] ||
                    { [ "${#_trimmed_id}" -eq "${#_n}" ] && [ "$_trimmed_id" -ge "$_n" ]; }; then
                    _bad=1
                else
                    _norm_ids+=("$((10#$_id))")
                fi
            done
            _dup=$(printf '%s\n' "${_norm_ids[@]}" | sort -n | uniq -d | paste -sd ',')
            if [ "$_bad" -eq 1 ]; then
                echo "WARNING: HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES is not a" \
                     "valid selection of ordinals 0..$((_n - 1)) -- the only" \
                     "ordinals ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES exposes" \
                     "after filtering -- or it has a stray leading, trailing, or" \
                     "doubled comma. This usually means a physical device ID was" \
                     "copied in verbatim, or a separator was left over from" \
                     "editing the list; left unchanged, so Ray will see no usable" \
                     "GPU. Unset HIP_VISIBLE_DEVICES and re-run this snippet, or" \
                     "correct it to ordinals in that range with no stray" \
                     "separators." >&2
            elif [ -n "$_dup" ]; then
                echo "WARNING: HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES repeats" \
                     "ordinal(s) $_dup once leading zeros are normalized -- Ray" \
                     "sizes its advertised GPU resources from the length of this" \
                     "mask, so a repeated ordinal creates two schedulable slots" \
                     "that both map to the same ROCr-visible device, letting" \
                     "concurrent ClientApps collide on it. Left unchanged, so" \
                     "remove the duplicate and re-run this snippet." >&2
            fi
            unset _hip_ids _id _bad _dup _norm_ids _trimmed_id
        fi
        unset _n
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
    when ``HIP_VISIBLE_DEVICES`` is unset: an existing value may deliberately select a
    subset of the renumbered space -- for example, to dedicate only one of two
    ROCr-visible devices to this process -- so it is not overwritten. But it is not
    assumed to already be expressed in that renumbered space either: the snippet checks
    that every ordinal it names is ``< n``, and warns instead of silently proceeding
    when one is not, since that is the signature of a physical device ID having been
    copied in rather than a deliberately chosen ordinal. It also rejects a repeated
    ordinal (``HIP_VISIBLE_DEVICES=0,0``), including one disguised by a leading zero
    (``HIP_VISIBLE_DEVICES=0,00``) -- the snippet normalizes each value before comparing
    rather than comparing the strings as spelled: Ray derives the number of GPU slots it
    advertises from the length of this mask, so a duplicate would let two schedulable
    slots map to the same underlying device instead of each naming a distinct one. The
    same reasoning applies to a trailing comma (``HIP_VISIBLE_DEVICES=0,1,``): Bash's
    ``read -ra`` silently drops the resulting empty field, but Ray splits the raw string
    itself and would still size its GPU resources from it, so the snippet checks the raw
    string for a trailing comma rather than trusting what ``read -ra`` parsed.

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
    concurrently. If you need them to run concurrently, start a separate managed local
    SuperLink for each -- a different ``FLWR_LOCAL_SUPERLINK_HTTP_API_PORT`` (and a
    different ``FLWR_HOME`` so they do not share state) per shell, exported before that
    shell's first ``flwr run`` -- so each SuperLink copies its own shell's
    ``HIP_VISIBLE_DEVICES`` at startup instead of all runs competing for one singleton.
    See :doc:`how-to-run-flower-locally` for how those variables control the managed
    SuperLink's address and storage location.

.. important::

    The guidance above assumes the single-node, Flower-managed-SuperLink flow. Flower
    also supports :ref:`multi-node simulations <multinodesimulations>`, where you run
    ``ray start --head`` on one node and ``ray start --address=...`` on each worker node
    yourself, before ``flwr run``. There, restarting the SuperLink does not help at all:
    each node's own ``ray start`` already performed AMD GPU discovery using whatever
    ``ROCM_PATH``, ``HIP_VISIBLE_DEVICES``, and ``ROCR_VISIBLE_DEVICES`` were present in
    *that node's* shell at the moment ``ray start`` ran there. Exporting corrected
    values afterwards -- in the driver shell that later runs ``flwr run``, or by
    restarting only the Flower-managed SuperLink -- changes nothing for nodes Ray itself
    started, since Flower does not manage those processes.

    For a manually started Ray cluster, ``ROCM_PATH`` and the
    ``HIP_VISIBLE_DEVICES``/``ROCR_VISIBLE_DEVICES`` handling above must therefore be
    exported in the shell on **every** node -- head and each worker -- *before* that
    node's ``ray start`` command runs. If a cluster is already running with the wrong
    environment, stop it (``ray stop`` on every node, or tear the cluster down and
    restart it) so each node redoes GPU discovery with the corrected environment; those
    nodes will keep reporting ``GPU=0`` until they are individually fixed and restarted.

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

This error is raised for two different reasons, and the warning immediately above it
tells you which one applies. Despite the error message, the process does not always
exit: the run can appear to hang after the error is printed. If that happens, interrupt
it and scroll back for the warning that precedes it.

No GPU reported while a GPU was requested
-----------------------------------------

If the preceding warning reads ``Ray reports no GPU resources (GPUs=0) but each client
requests num_gpus=...``, Ray detected no GPU at all while your ``ClientApp`` requested a
fraction of one:

.. code-block:: text

    WARNING : Ray reports no GPU resources (GPUs=0) but each client requests
    num_gpus=0.25, which no node can satisfy. This does not by itself mean GPU
    discovery failed: the cluster may have no GPU, or Ray's GPU resources may
    have been limited deliberately (for example `--init-args-num-gpus 0`). Set
    `num_gpus=0` if no GPU is expected, otherwise review that limit before
    suspecting discovery.

Three different situations produce this warning, and only one of them calls for the ROCm
fixes elsewhere in this guide:

- The cluster genuinely has no GPU. Set ``client-resources-num-gpus`` to ``0`` and move
  on.
- Ray's GPU resources were limited deliberately, for example via ``flwr federation
  simulation-config --init-args-num-gpus 0``. Lowering ``client-resources-num-gpus``
  will not help here either -- no non-zero value can be satisfied by a cluster reporting
  ``GPUs=0`` -- but neither will ``ROCM_PATH`` or ``HIP_VISIBLE_DEVICES``, since the
  limit is applied at ``ray.init()`` and never reaches ROCm. Review, or raise, the
  configured ``init-args-num-gpus`` value instead.
- ROCm reaches the GPU but Ray did not discover it. Work through `Point Ray at the ROCm
  SMI library`_ and `Set HIP_VISIBLE_DEVICES for Ray`_, then re-run the verification
  above.

Compare what Ray actually detects (``ray.cluster_resources()``, as verified above)
against the ``--init-args-num-gpus`` value you configured to tell the second case apart
from the third.

If the cluster also has too few CPUs for even one client, Flower emits an additional
warning right after this one, naming that CPU shortage explicitly -- in that case,
setting ``num_gpus=0`` alone will not be enough; also lower
``client-resources-num-cpus`` or add CPU capacity to the cluster.

If ``/dev/kfd`` exists, Flower follows the warning above with a second one that depends
on what else it can detect about device access:

- If the process cannot open ``/dev/kfd`` -- the missing ``render``-group access covered
  in `Check that ROCm can reach the GPU`_ -- Flower reports a permissions problem and
  states explicitly that fixing ``ROCM_PATH`` will not help; follow that warning's
  instructions instead, not the ``ROCM_PATH`` steps below.
- If ``/dev/kfd`` is accessible but no ``/dev/dri/renderD*`` node is accessible at all,
  Flower reports that render-node gap instead -- again stating that ``ROCM_PATH`` will
  not help, since ROCm needs both device nodes. Fix render-node access the same way as
  ``/dev/kfd`` access above.
- If ``/dev/kfd`` and a render node are both accessible, but that render node is not
  AMD-backed -- for example a hybrid host or container exposing an integrated Intel
  GPU's ``renderD128`` while the AMD device's own render node is missing or
  permission-denied -- Flower reports that vendor mismatch instead, again stating that
  ``ROCM_PATH`` will not help. Fix render-node pass-through/permissions so that an
  AMD-backed node specifically (identified by its sysfs PCI vendor ID) is accessible,
  not just any render node.
- Only when both ``/dev/kfd`` and an AMD-backed render node are accessible does Flower
  name the ``ROCM_PATH`` requirement directly; in that case, follow `Point Ray at the
  ROCm SMI library`_.

Note that ``torch.cuda.is_available()`` commonly reports ``True`` in exactly this
situation, because your ML framework locates HIP without consulting ``ROCM_PATH``.

CPU or GPU capacity is too small for even one client
----------------------------------------------------

If the preceding warning instead reads ``The ActorPool is empty. The system (CPUs=...,
GPUs=...) does not meet the criteria to host at least one client...``, Ray *did* detect
resources, but no single node has enough of them for a client on its own -- for example
a node reporting ``CPU=1`` with ``client-resources-num-cpus`` set to ``2``, or a 1-GPU
node with ``client-resources-num-gpus`` set to ``2``. This is a resource-sizing problem,
not a GPU discovery problem: lower ``client-resources-num-cpus`` and/or
``client-resources-num-gpus``, or add CPU/GPU capacity to the cluster. The ``ROCM_PATH``
and ``HIP_VISIBLE_DEVICES`` fixes elsewhere in this guide do not apply here.

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
