# Frozen evaluation artifacts

This directory keeps only compact files needed to identify, audit and reproduce
decision-relevant runs. Datasets, checkpoints, expanded `raw.log`, videos and
per-episode traces are not committed.

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

Compact ApexTarget V1/V2 artifacts and the paired ProbeSet summaries are kept
alongside the base runs for reproducible comparison. OF-base remains the default
policy; ApexTarget and all later development lines are experiment-only.

The machine-readable archive index is
`archive/20260828_retired_research/retired_research_summary_v1.json`. Detailed
per-episode structured evidence is retained locally as the untracked archive
`structured_episode_evidence.tar.gz`; its checksum and the immutable code tags
are recorded in
[`docs/RETIRED_RESEARCH_ARCHIVE_20260828.md`](../docs/RETIRED_RESEARCH_ARCHIVE_20260828.md).

See also the combined full-run paired audit in
[`docs/OF_BASE_APEXTARGET_V1_V2_FULL_AUDIT.md`](../docs/OF_BASE_APEXTARGET_V1_V2_FULL_AUDIT.md).
