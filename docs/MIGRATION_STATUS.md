# ZSON3 migration status

Baseline source: OpenFrontier commit
`a3f8b83da6135a88247651534061df2ea05850f6`, tagged locally as
`openfrontier-upstream-a3f8b83`.

Active branch: `zson3-runtime-0.3.3`.

## Change classification

All changes in this first batch are **runtime migration** or test
instrumentation. No FrontierManager, frontier association, information-gain,
VLM prompt/parser, utility, target approach, STOP, or recovery decision rule
has been changed.

## Passed gates

### Runtime gate

- Habitat-Sim/Lab/Baselines 0.3.3 import successfully in `zson3`.
- HM3D ObjectNav v1 `val` is loaded through project-local data symlinks.
- Fixed scene `6s7QHgap2fW`, episode `0` resets and steps successfully.
- RGB, depth, GPS, compass, ObjectGoal and evaluation metrics are present.
- Frozen comparison protocol is in `config/zson3/hm3dv1_val.yaml`.
- A shared dataset adapter now resolves HM3Dv1, HM3Dv2, and MP3D without
  placing dataset-specific paths inside OpenFrontier algorithm code.
- One fixed scene from each dataset successfully instantiates and resets under
  Habitat 0.3.3 with the same RGB/depth/GPS/compass/ObjectGoal contract.
- `hm3d` remains the HM3Dv1 compatibility alias. HM3Dv1/v2 use the frozen
  0.1 m success radius; MP3D preserves upstream OpenFrontier's 1.0 m radius.

The upstream benchmark's HM3D-v2 path and synthetic per-scene split are
replaced by a standard HM3Dv1 `val` split plus `content_scenes` filtering.
Habitat success distance is 0.1 m for comparability with the standard HM3Dv1
protocol. OpenFrontier's internal object-approach threshold remains 1.0 m.

### Frontier component gate

- FrontierNet checkpoint SHA256:
  `5b28d0ed3a921fe0899a7b1d1c86dd25e17faa628270737fd163a31451508aa6`.
- Fixed input: `6s7QHgap2fW`, episode `0`, six `turn_right` actions.
- CPU trace: 1589 frontier pixels and three anchored 3D frontiers.
- Stable input-array and output hashes are recorded in
  `config/zson3/frontier_fixture_turn6.json`.

The checkpoint loader no longer downloads ImageNet ResNet34 weights before
strictly loading the complete FrontierNet checkpoint. Network topology and
final model parameters are unchanged.

### Mapping component gate

- The upstream `WaveMapper` integrates the fixed depth frame successfully.
- A bounded 35,301-point local query returns occupied and free evidence.
- The full upstream global interpolation is intentionally deferred to the
  episode gate because it constructs a much larger fixed query lattice.

### PointNav component gate

- The existing VLFM checkpoint is reused through a project-local symlink;
  SHA256 is
  `ecb6f217fad7abed04dea5db36f1a88cf1d49e58943be4d283ba3de64c2ac2c2`.
- Its pre-0.3 flat policy config is upgraded in memory to the Habitat-Baselines
  0.3.3 `main_agent` layout. Network topology and checkpoint tensors are not
  rewritten.
- All 80 state-dict entries load strictly; partial weight loading is forbidden.
- A deterministic CUDA forward using the real 224x224 depth/pointgoal contract
  returned a valid discrete action and recurrent state shape `[1, 4, 512]`.

### Semantic component gate

- Local Qwen3-VL-8B health and generation endpoints are executable through a
  proxy-independent client.
- The fixed turn-6 frontier image produces the same A/B/C set-of-marks input.
- The upstream chair/A-B-C prompt remains byte-identical; SHA256 is
  `5d899ff8772232783ddd43a22f80f894644bf99de1306f748af180dbdb577ba7`.
- Qwen returned valid A/B/C probability/reason pairs and the unchanged
  OpenFrontier JSON semantics parsed them successfully.
- The first observed request latency was 61.1 seconds. The reused service had
  been loaded with `device_map=auto`, and its startup log confirmed CPU
  offload. A same-prompt, same-image rerun after forcing the complete model
  onto GPU1 (`cuda:0` inside `CUDA_VISIBLE_DEVICES=1`) took 6.52 seconds, a
  9.4x reduction, with valid JSON output.

The Qwen backend is an explicit ZSON3 configuration variant in
`config/zson3/navigation_hm3dv1_qwen.yaml`; upstream `config/navigation.yaml`
is not overwritten. This establishes an executable local backend, not numeric
or performance equivalence to OpenFrontier's Gemini results.

Project scripts now health-check/reuse the existing Qwen process, start it only
when absent, and stop it only when a ZSON3-owned PID record exists. Machine-wide
HTTP proxies are bypassed for loopback model traffic.
The default service device map is explicitly full-GPU and can be overridden
with `ZSON3_QWEN_DEVICE_MAP`.

For evaluation, the default backend is now the isolated vLLM 0.15.1 service
with Torch 2.9.1/cu128. The fixed semantic fixture took 1.59 seconds after
warmup, compared with 6.20 seconds for full-GPU Transformers plus
FlashAttention 2. Backend selection changes runtime only; the OpenFrontier
prompt and response parser are shared. Exact environment and fallback commands
are recorded in `docs/QWEN_RUNTIME.md`.

### Random-100 evaluation gate

`scripts/run_hm3dv1_random100.py` reproduces Habitat's seeded sampling order
with seed `20260727`, `shuffle=true`, `group_by_scene=true`, 100 sampled
episodes, and one episode per scene repetition. It freezes the resulting
scene/episode/target identities to `manifest.json` before navigation starts.

The runner writes one JSON result per episode, cumulative `summary.json` and
`summary.txt`, and supports a contiguous-prefix resume. The shell entry point
`scripts/run_openfrontier_random100.sh` additionally creates append-only
`raw.log` and concise `progress.log` files. Required model services are checked
before every episode, and transport loss aborts the run without adding the
interrupted episode to the resumable completed prefix. Time-limit enforcement
does not emit a Habitat STOP action, so it cannot create an accidental success.

The first attempted random-100 run is retained only as a runtime-failure
artifact: its temporary interactive vLLM process disappeared during episode 3,
and later episodes were dominated by retry waits. Those results are excluded
from performance analysis. The corrected default output directory is
`results/openfrontier_random100_runtimefix_seed20260727/`.

### Fixed episode gate

The complete migrated stack passed one frozen HM3Dv1 `val` episode:

- scene `6s7QHgap2fW`, episode `0`, target `chair`, seed `0`;
- termination reason `object_found` after 194 navigation steps;
- success `1.0`, SPL `0.4597688044669626`, final distance to goal
  `0.028764251619577408` m;
- elapsed time 1285.8 seconds and process peak RSS 3942.4 MiB;
- 31 Habitat collision events, without a crash, deadlock, or stale-process
  failure.

This is an executable-closure gate, not a performance estimate. The trace
exercised the full runtime chain rather than isolated substitutes:

```text
Habitat 0.3.3 RGB-D/GPS/Compass
  -> WaveMapper + FrontierNet
  -> FrontierManager + information gain
  -> local Qwen frontier probabilities
  -> adapted PointNav policy -> Habitat discrete action
  -> SAM3 text-prompt masks
  -> DetectedObject association and viewpoint approach
  -> local Qwen target verification (0.95)
  -> locked-in object approach -> STOP
```

The run made 19 frontier-scoring calls and one target-verification call. Qwen
accounted for 1177.9 seconds across the 20 calls; SAM3 accounted for 55.8
seconds across 32 calls. Navigation excluding AI and mapping took 15.2 seconds.
Those episode timings came from the CPU-offloaded service and are not
representative of the corrected full-GPU deployment. A subsequent episode gate
must remeasure end-to-end latency before deciding whether a dedicated inference
engine is necessary.

The reproducible entry point is `scripts/run_fixed_hm3dv1_episode.py`. It
records metrics, component timings, termination reason, peak process RSS and
exceptions under an ignored artifact directory. `--max-steps` is consumed by
the unchanged agent; the runner independently enforces `--max-time`, because
the upstream agent's internal time-limit block is commented out.

With the local Qwen and SAM3 services healthy, reproduce the gate on GPU0 with:

```bash
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet CUDA_VISIBLE_DEVICES=0 \
  conda run --no-capture-output -n zson3 \
  python scripts/run_fixed_hm3dv1_episode.py \
  --output-dir artifacts/runs/hm3dv1_6s7QHgap2fW_ep0
```

## Known warnings and boundaries

- Habitat emits duplicate Magnum plugin warnings in this editable-install
  setup.
- The HM3D scene-dataset manifest references train/test scenes not installed on
  this server, producing missing-glob warnings. The selected val scenes load.
- The committed fixture identity is based on array hashes. Generated `.npz`
  containers are ignored by git and their archive-byte hash is not a protocol
  identifier.
- CPU FrontierNet output is frozen. CUDA numerical parity remains to be
  measured before using a GPU trace as a golden output.
- A single end-to-end episode is confirmed, but it is insufficient for SR/SPL
  or robustness claims. A fixed multi-episode evaluation is still required.

### Segmentation closure audit

The pinned SAM3 submodule is now initialized at `5dd401d`, but no SAM3
environment or gated `facebook/sam3/sam3.pt` checkpoint is available locally.
Port 12184 also conflicts with the existing YOLOv7 service. The exact external
gate and isolated environment commands are recorded in
`docs/SEGMENTATION_CLOSURE.md`; ZSON3 reserves port 12186 for the future SAM3
adapter. Existing bbox-prompted MobileSAM is explicitly not treated as SAM3.

The isolated `zson3-sam3` environment imports successfully with CUDA 12.8 and
the 12186 server/client adapter is implemented. A compatible `sam3.pt` from the
user-provided `1038lab/sam3` mirror loaded successfully; SHA256 is
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
The fixed HM3Dv1 frame produced a valid non-empty text-prompt segmentation.
Official repository access is still pending, so direct official/mirror hash
identity is not yet proven.

## Next bounded migration batch

1. Freeze a small deterministic HM3Dv1 episode manifest and run it without
   changing algorithms, recording termination, SR/SPL, collisions, component
   timings, RSS, crashes, and stale state.
2. Compare the 0.3.3 PointNav adapter with the legacy VLFM action oracle on a
   fixed observation trace before claiming action parity.
3. Preserve this successful trace while investigating repeated Qwen scoring
   latency and the observed high collision count. Neither observation is yet
   evidence for an algorithm change.
4. Validate HM3Dv2 and MP3D through complete episodes after the HM3Dv1 baseline
   protocol is frozen; their reset/observation contract has already passed.
