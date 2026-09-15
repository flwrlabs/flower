# Flower end-to-end tests

This directory contains folders for different scenarios that need to be tested and
validated before a change can be added to Flower.

## Flux SuperNode test

Install the `e2e-bare` app and run the Flux SuperNode test from its directory:

```bash
cd framework/e2e/e2e-bare
python -m pip install --upgrade .
../test_supernode_flux.sh
```

By default, the test uses the bundled Flux command fixture. The fixture implements the
`flux run` interface used by the deployment guide, so the test can run in CI without a
Flux installation. To exercise the same test against an active Flux instance, set the
Flux executable explicitly:

```bash
FLWR_E2E_FLUX_BIN="$(command -v flux)" ../test_supernode_flux.sh
```

The test starts a local SuperLink, launches two SuperNodes through `flux run`, submits
the `e2e-bare` Flower App, and waits for the run to finish. It allocates unused local
ports and removes its temporary Flower state when it exits.
