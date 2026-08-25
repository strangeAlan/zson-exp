# Frozen evaluation artifacts

This directory keeps only lightweight files needed to identify and reproduce
important full-split runs. Datasets, checkpoints, `raw.log`, per-episode JSON
traces and visual diagnostics remain local.

## OF-base frozen full runs

`openfrontier_base_sam3_full_hm3dv1_2000_seed20260727/` contains the completed
HM3Dv1 2000-episode manifest and lightweight logs. Its frozen evaluator uses
the original 0.1 m protocol (50.00% SR / 0.2503 SPL). For the common target
module audit, `audit_1m.json` reconstructs 54.00% SR@1m / 0.2747 SPL@1m from
the same trajectories; no episode was rerun.

`openfrontier_base_sam3_full_hm3dv2_1000_seed20260727/` contains the frozen
1000-episode manifest, evaluator contract, service metadata, progress heartbeat
and final summaries for OpenFrontier + SAM3 + Qwen:

- official SR@1m: **70.80%**;
- official SPL@1m: **0.3299**;
- diagnostic SR@0.1m / SPL@0.1m: 39.50% / 0.1821;
- exceptions: 0.

The corresponding ApexTarget artifacts remain on the
`apextarget-experimental` branch. Immutable rollback tags and the combined
paired conclusions are documented in
[`docs/OF_BASE_APEXTARGET_V1_V2_FULL_AUDIT.md`](../docs/OF_BASE_APEXTARGET_V1_V2_FULL_AUDIT.md).
