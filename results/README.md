# Published evaluation artifacts

Only lightweight, reproducibility-relevant files from the completed full runs
are versioned here:

- `manifest.json`: frozen episode identities and order;
- `progress.log`: one compact heartbeat per completed episode;
- `summary.json` / `summary.txt`: aggregate metrics;
- `services.json`: model-service runtime metadata;
- V1 `floor_summary.*`: offline evaluator-only floor grouping summary.

Raw logs, per-episode traces, images and checkpoints are intentionally omitted.
They remain local because the complete result directory is several gigabytes.

| Run | Primary result |
| --- | --- |
| `openfrontier_apextarget_v1_deterministic_full_hm3dv1_2000_seed20260727` | SR 50.45%, SPL 0.2300 at 0.1 m |
| `openfrontier_apextarget_v1_deterministic_full_hm3dv2_1000_seed20260727` | SR 65.20%, SPL 0.2790 at the official 1 m radius |

The HM3Dv2 `summary.json` preserves both radii. Its primary fields are
`sr_at_1m` and `spl_at_1m`; `sr` and `spl` are the retained 0.1 m diagnostic.

See `docs/HM3DV1_FULL_AUDIT.md` and `docs/HM3DV2_FULL_AUDIT.md` for analysis
and interpretation boundaries.
