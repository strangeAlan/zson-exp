# OpenFrontier Base SR55 Random-100 Seal

This directory freezes the first completed OpenFrontier-derived ZSON3
HM3Dv1 random-100 baseline.

## Identity

- Source branch: `zson3-runtime-0.3.3`
- Evaluated code commit: `6b8d3f4eae5afe033b93b0f9348586393b956e29`
- Dataset: HM3Dv1 val
- Sampling seed: `20260727`
- Episodes: 100
- Runtime: Habitat-Sim / Habitat-Lab 0.3.3
- Perception services: SAM3 and local Qwen through vLLM

The annotated Git tag `openfrontier-base-sr55-random100-seed20260727`
points to the commit that adds this seal. The algorithm/runtime evaluated is
the parent commit recorded above; the seal commit changes no runtime code.

## Result

- Successes: 55 / 100
- SR: 55.00%
- SPL: 0.2531794480
- Elapsed time: 22533.17 seconds
- Exceptions: 0
- Process exit code: 0

## Contents

- `manifest.json`: selected episode manifest and protocol arguments.
- `summary.json`, `summary.txt`: final metrics.
- `progress.log`: one-line episode progress and final completion record.
- `services.json`: service endpoints captured by the runner.
- `episodes.tar.gz`: all 100 structured episode result JSON files.
- `episode_logs.tar.gz`: all per-episode stdout/stderr/combined diagnostic logs.
- `SHA256SUMS`: checksums for every sealed payload.

The 15 MB aggregate `raw.log` is not duplicated because its evidence is
already retained in the structured results and per-episode logs.

## Integrity checks performed before sealing

- Exactly 100 episode JSON files exist.
- Exactly 100 episode log groups exist.
- Episode indices cover the completed manifest without runner exceptions.
- Final summary reports 100 episodes, 55 successes, and zero exceptions.
- The `failures/` directory contains no service/runtime failure artifact.

Run `sha256sum -c SHA256SUMS` in this directory before using the artifact.
