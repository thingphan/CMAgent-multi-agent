# CellMarker v2.6 strict entry point

Run `run_pipeline.py` from the repository root, or use the Docker image. See the
[main README](../../../README.md) for setup, input contracts, and upstream integration.

The relative symlinks in this directory expose the canonical agent/controller
implementations without duplicating source files. Keep their targets under `src/`
when packaging the code.

Defaults: zero-shot extraction, dictionary hints enabled, selective verification,
local mapping followed by cell and gene normalization agents. Positive, negative,
and uncertain relation polarities are preserved.
