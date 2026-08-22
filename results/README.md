# Frozen evaluation artifacts

This directory keeps only lightweight files needed to identify and reproduce
important full-split runs. Datasets, checkpoints, `raw.log`, per-episode JSON
traces and visual diagnostics remain local.

## HM3Dv2 pre-Apex main baseline

`openfrontier_base_sam3_full_hm3dv2_1000_seed20260727/` contains the frozen
1000-episode manifest, evaluator contract, service metadata, progress heartbeat
and final summaries for OpenFrontier + SAM3 + Qwen:

- official SR@1m: **70.80%**;
- official SPL@1m: **0.3299**;
- diagnostic SR@0.1m / SPL@0.1m: 39.50% / 0.1821;
- exceptions: 0.

The corresponding ApexTarget artifacts remain on the
`apextarget-experimental` branch. See
[`docs/HM3DV2_PREAPEX_PAIRED_AUDIT.md`](../docs/HM3DV2_PREAPEX_PAIRED_AUDIT.md)
for the strict paired analysis.
