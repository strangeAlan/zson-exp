# OpenFrontier Base Random-100 Analysis

## Scope and evidence boundary

This report compares the sealed OpenFrontier-derived ZSON3 baseline with the
sealed VLFM H1 (`hybrid_coherent`) and T1 (`t1_apex_fusion_medoid`)
random-100 runs. It deliberately excludes multi-floor work.

The current ZSON3 run and the VLFM runs share the nominal seed `20260727`, but
they do **not** contain the same sampled episodes. Only 12 of 100
`(scene, episode_id)` identities overlap. H1 and T1, by contrast, use exactly
the same 100 identities. Therefore aggregate ZSON3-versus-VLFM differences are
useful signals, not paired causal measurements.

## Aggregate comparison

| Run | Episodes | SR | SPL | False positives | Other failure labels |
| --- | ---: | ---: | ---: | ---: | --- |
| ZSON3 OpenFrontier base | 100 | 55% | 0.2532 | 20 | 23 max steps, 2 stuck |
| VLFM H1 hybrid coherent | 100 | 55% | 0.2186 | 20 | 3 FN, 20 never-saw/search failures, 2 stair-labelled failures |
| VLFM T1 Apex fusion medoid | 100 | 55% | 0.2189 | 18 | 4 FN, 21 never-saw/search failures, 2 stair-labelled failures |

The ZSON3 SPL is 0.0343 above H1 and 0.0343 above T1, about a 15.7% relative
difference. This suggests that successful ZSON3 trajectories may be more
efficient, but the different sample, evaluator, controller, detector, and
runtime prevent attributing that difference to OpenFrontier itself.

H1 and T1 provide a cleaner warning about target fusion: T1 reduces false
positives from 20 to 18, but false negatives rise from 3 to 4. At the episode
level, T1 rescues two H1 failures and loses two H1 successes, leaving SR
unchanged. A stricter target gate alone is therefore not a demonstrated gain.

## Current ZSON3 failure structure

Of the 45 failures:

- 23 (51.1%) reach the episode step limit.
- 20 (44.4%) stop on a false positive.
- 2 (4.4%) end in the current stuck condition.

The false positives are not mainly a small STOP-radius mismatch. Their final
distance to the nearest ground-truth goal has mean 5.16 m and median 3.14 m:

- 11 / 20 are at least 3 m away.
- 6 / 20 are between 1 m and 3 m away.
- 1 / 20 is between 0.5 m and 1 m away.
- 2 / 20 are below 0.5 m.

Thus 17 / 20 false positives finish more than 1 m from the target. This is
direct evidence of a target perception, association, verification, or object
goal selection problem, rather than only an approach-controller problem.

The strongest class-localized signal is sofa:

| Target | Episodes | Successes | SR | False positives | Max steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| bed | 24 | 12 | 50.0% | 2 | 10 |
| chair | 23 | 15 | 65.2% | 5 | 2 |
| sofa | 22 | 11 | 50.0% | 9 | 2 |
| toilet | 19 | 11 | 57.9% | 2 | 5 |
| tv/monitor | 11 | 5 | 45.5% | 2 | 4 |
| plant | 1 | 1 | 100% | 0 | 0 |

Nine of the 11 sofa failures are false positives. The single plant episode is
not statistically meaningful. Bed and tv/monitor also expose search coverage
or false-negative risk through their max-step failures.

Among the 23 max-step failures, 13 end with no detected object in memory and
10 end with at least one detected object. Existing logs show candidate masks
and rejected verification in some of the latter group, but do not record
ground-truth target visibility. We therefore cannot yet distinguish detector
false negatives from correct rejection of irrelevant masks or exploration
that never exposed the target.

When the selected target is correct, goal completion is precise: all 55
successes end within roughly 6.2 cm of a valid goal. The two stuck failures are
too few to justify prioritizing recovery over target perception.

## Runtime signal

Across the 100 episodes, recorded model/runtime totals include:

- SAM: 6722.65 s over 3757 calls, 1.789 s/call.
- Qwen VLM calls: 4277.35 s over 2341 calls, 1.827 s/call.
- Mapping updates: 3086.29 s over 3827 calls, 0.806 s/call.
- PointNav planning: 601.22 s over 23198 calls, 0.026 s/call.

These timers can overlap or be nested and must not be summed as wall time.
They do establish that SAM is a first-order latency source, while PointNav
planning is not the immediate runtime bottleneck.

## Decision

The 100 episodes are sufficient to identify target handling as a concrete
weakness, especially cross-view confidence and sofa false positives. They are
not sufficient to claim that ZSON3 is better than H1/T1 overall, because the
episode manifests and execution protocols differ.

The next sequence should be:

1. Keep this tag as the immutable pre-optimization baseline.
2. Optimize SAM transport/inference latency without changing its output
   semantics, and verify equivalence on a fixed trace.
3. Add target-side observability needed by a full run: track identity,
   per-view confidence, positive/negative verifier evidence, visibility,
   selected approach point, and STOP-to-ground-truth distance. This is
   instrumentation, not an algorithm change.
4. Run the full HM3Dv1 baseline once at the faster runtime.
5. Use the full-run failure distribution to define a bounded ApexNav-style
   target subsystem ablation: cross-frame association, confidence fusion,
   positive/negative evidence, visibility-aware decay, multi-view
   confirmation, and robust approach-point selection.

Do not port ApexNav's ROS runtime, do not merely raise the verifier threshold,
and do not begin upstairs work at this stage. The H1/T1 paired result already
shows that trading two false positives for extra false negatives can leave SR
unchanged.
