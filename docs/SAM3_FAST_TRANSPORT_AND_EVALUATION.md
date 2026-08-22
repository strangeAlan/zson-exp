# SAM3 Fast Transport and Evaluation Protocol

## Baseline boundary

The immutable pre-optimization baseline remains:

- Tag: `openfrontier-base-sr55-random100-seed20260727`
- Evaluated runtime commit: `6b8d3f4eae5afe033b93b0f9348586393b956e29`
- Result: SR 55%, SPL 0.2532

The changes after that tag are runtime transport and evaluator
instrumentation. They do not alter OpenFrontier frontier generation,
information gain, frontier utility, target thresholds, model weights, prompts,
or action selection.

## SAM3 optimization

The old localhost protocol expanded RGB bytes and mask pixels into JSON lists.
The optimized protocol uses:

- lossless contiguous uint8 RGB encoded with base64;
- lossless boolean masks encoded with NumPy `packbits` and base64;
- `torch.inference_mode()` around the existing bfloat16-autocast inference;
- no periodic `torch.cuda.empty_cache()` or Python GC in the request path.

TF32, quantization, resized input, changed thresholds, and `torch.compile` are
not enabled.

The one-shot fixed fixture gate is:

```bash
/home/hsy/miniconda3/envs/zson3/bin/python \
  scripts/verify_sam3_transport.py \
  --fixture artifacts/fixtures/hm3dv1_6s7QHgap2fW_ep0_turn6.npz \
  --prompt chair
```

Observed after warmup:

- boxes and scores were float32-exact;
- both protocols returned zero masks for this fixed view and prompt;
- the separate non-empty synthetic codec gate was bit-exact;
- request JSON decreased from 4,806,789 to 1,228,940 bytes;
- server inference remained 0.161 versus 0.158 seconds;
- localhost round trip decreased from 0.622 to 0.175 seconds.

A 12-step integrated HM3D smoke made two SAM composition calls in 0.460
seconds total and completed without an exception.

## Target diagnostics

Episode results now retain evaluation-only evidence needed for full-run failure
analysis:

- per-step ground-truth target semantic visibility and pixel fraction;
- each segmentation event and candidate mask count;
- candidate SAM confidence, box, image index, viewpoint, and 3D centroid;
- persistent object track ID, first/last step, and observation count;
- Qwen verification probability, threshold, decision, and reason;
- selected object approach path endpoint and path length;
- final object tracks and termination object geometry.

The Habitat semantic sensor is used only to write visibility diagnostics. Its
output is not exposed to the navigation policy.

## T1-matched random-100

The manifest
`config/evaluation/hm3dv1_t1_random100_seed20260727.json` was derived from the
sealed T1 episode archive. Habitat 0.3.3 resolved all 100 identities, and the
resolved `(scene_id, episode_id)` set is exactly equal to T1.

The compact T1 progress log does not preserve the identities of successful
episodes in execution order. Therefore this manifest uses canonical
scene/episode order. The evaluated set is identical; only ordering is not
claimed to match.

Run in tmux:

```bash
tmux new-session -d -s zson3-t1-r100 \
  'cd /home/hsy/zson-exp && bash scripts/run_openfrontier_t1_random100.sh'
tmux attach -t zson3-t1-r100
```

Progress is written immediately to:

```text
/home/hsy/zson-exp/results/openfrontier_t1_exact_random100_samfast_seed20260727/progress.log
```

## Complete HM3Dv1 val

The full manifest gate resolved 2,000 unique validation episodes.

Run only after reviewing the matched random-100 result:

```bash
tmux new-session -d -s zson3-full-v1 \
  'cd /home/hsy/zson-exp && bash scripts/run_openfrontier_full_hm3dv1.sh'
tmux attach -t zson3-full-v1
```

Progress is written immediately to:

```text
/home/hsy/zson-exp/results/openfrontier_base_full_hm3dv1_samfast_seed20260727/progress.log
```

Both launchers are resumable and abort on a required Qwen or SAM3 service
failure instead of silently completing a corrupted episode.
