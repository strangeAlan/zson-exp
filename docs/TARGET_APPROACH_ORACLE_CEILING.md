# Target Approach Oracle Ceiling final audit

Date: 2026-08-28

Frozen baseline: `of-base-full-v1-v2-20260727` (`3a79d975`)

Analysis branch: `analysis/target-approach-oracle`

Implementation commit: `2d804ab`
Result: **NO-GO — stop the target grounding / approach patch line**

## Executive conclusion

The experiment answers the requested ceiling question negatively.

Among the eight fixed OF-base `accepted-but-no-explicit-STOP` failures, five
accepted candidates are correct, two are ambiguous because the selected source
frame has no GT target pixels, and one is clearly wrong. Candidate-derived
Oracle A rescues only **1/5 correct-candidate failures (20%)**, below the 50%
continuation threshold. It rescues one of all eight failures (12.5%).

The four correct-candidate Oracle-A failures all fail again when Oracle B sends
the unchanged PointNav directly to the nearest reachable dataset success
viewpoint: **0/4 additional rescues**. Thus these failures cannot be explained
by surface reconstruction or approach-viewpoint sampling. The current PointNav
executor is the binding ceiling in these cases.

The ten ordinary OF-base protection replays have **zero losses**. All ten match
the frozen full run exactly in success, reason, navigation steps, SPL, and first
accepted step/centroid/probability. The analysis infrastructure therefore did
not perturb the protected OF-base path.

The predeclared decision rule was:

> GO only if Oracle A rescues at least half of correct-candidate
> accepted-no-STOP failures and protection loss is zero.

Observed: `1/5 = 20%` and protection loss `0/10`. The final decision is
**NO-GO**, because the rescue condition fails decisively. Oracle B provides no
countervailing evidence.

For completeness, SR@1m / SPL@1m are:

- all 11 OF-base failures in the diagnostic set under A: 18.18% / .0425;
- all eight accepted-no-STOP failures under A: 12.5% / .0211;
- the five correct-candidate accepted-no-STOP failures under A: 20% / .0338;
- the four Oracle-B executor controls: 0% / 0;
- the complete mixed 13-episode A run, including two already-successful
  controls: 30.77% / .1381.

The mixed 13-episode number is an audit-run aggregate, not a deployable policy
score and not the decision denominator.

## Frozen experiment and policy isolation

The fixed manifest contains 13 unique diagnostic episodes and ten protection
episodes. It was committed before full execution. Both the ProbeSet index and
the original full-1000 index are stored for every episode, preventing the two
index spaces from being confused.

Before original global Qwen acceptance, Oracle A/B execute OF-base unchanged.
GT semantic masks are used only for offline candidate classification. GT goal
viewpoints are used only to rank candidate-derived endpoints in A or provide the
executor-control endpoint in B. Online termination never polls GT success.
Both Oracle modes use a uniform 300-step post-acceptance allowance; protection
uses the official 500-step horizon and ordinary `PointnavAgent`.

The frozen implementation and thresholds are documented in
[`TARGET_APPROACH_ORACLE_PROTOCOL.md`](TARGET_APPROACH_ORACLE_PROTOCOL.md).

## Candidate correctness

The accepted track's latest retained SAM3 evidence is compared with the target
semantic IDs in the same source frame. `P` is mask precision and IoU is mask/GT
IoU. The source step is often earlier than Qwen acceptance because OF-base
accepts a persistent merged object track from a multi-frame global window.

| Probe | Role | Target | Accepted / source step | Pred / GT / overlap pixels | P | IoU | Audit |
|---:|---|---|---:|---:|---:|---:|---|
| 12 | Stable-v0 rescue | tv | 70 / 65 | 2,176 / 8,130 / 2,013 | .925 | .243 | correct |
| 14 | Stable-v0 rescue | plant | 115 / 90 | 1,165 / 516 / 124 | .106 | .080 | ambiguous boundary overlap |
| 19 | Stable-v0 rescue | chair | 29 / 23 | 5,578 / 4,808 / 0 | 0 | 0 | wrong |
| 24 | accepted-no-STOP | chair | 185 / 182 | 3,084 / 0 / 0 | 0 | 0 | ambiguous, GT absent |
| 25 | accepted-no-STOP + rescue | plant | 234 / 209 | 77 / 5,048 / 65 | .844 | .013 | correct |
| 26 | accepted-no-STOP | bed | 51 / 12 | 6,963 / 29,523 / 6,517 | .936 | .217 | correct |
| 27 | accepted-no-STOP + rescue | tv | 79 / 77 | 1,096 / 5,022 / 1,096 | 1.000 | .218 | correct |
| 28 | accepted-no-STOP | toilet | 102 / 101 | 2,162 / 0 / 0 | 0 | 0 | ambiguous, GT absent |
| 29 | accepted-no-STOP | sofa | 56 / 51 | 2,445 / 7,242 / 0 | 0 | 0 | wrong |
| 30 | accepted-no-STOP | bed | 97 / 94 | 5,563 / 21,656 / 5,276 | .948 | .240 | correct |
| 31 | accepted-no-STOP | tv | 474 / 424 | 1,339 / 3,591 / 783 | .585 | .189 | correct |
| 32 | Stable-v0 regression control | tv | 21 / 17 | 3,804 / 15,960 / 3,597 | .946 | .222 | correct |
| 35 | Stable-v0 regression control | toilet | 26 / 18 | 2,522 / 14,604 / 2,426 | .962 | .165 | correct |

The unique failure attribution therefore contains five category-A cases:
three ambiguous cases (14, 24, 28) and two clear wrong-candidate cases (19,
29). Better endpoints cannot repair them.

## Oracle A: candidate-derived viewpoint ceiling

Every selected A endpoint is navmesh reachable and lies very close to a legal
GT success viewpoint: its endpoint-to-GT geodesic is at most 0.152 m in all six
correct-candidate OF-base failures. `Path` is the acceptance-pose-to-endpoint
navmesh distance. `rho` reports start/min/final. `F/T` counts FORWARD versus
combined left/right turns after acceptance.

| Probe | Target | Candidate viewpoints | Endpoint XYZ | Path / endpoint→GT (m) | Final GT (m) | rho start/min/final | F/T | Outcome |
|---:|---|---:|---|---:|---:|---|---:|---|
| 12 | tv | 79 | (3.22, -0.81, 1.02) | 1.30 / .101 | .102 | 1.10/.16/.16 | 8/9 | rescue |
| 25 | plant | 61 | (3.43, -8.24, .90) | 2.63 / .002 | .009 | 2.63/.11/.11 | 11/7 | rescue |
| 26 | bed | 10 | (-7.40, -2.51, 3.97) | 1.83 / .152 | 1.182 | 1.56/1.54/1.60 | 16/23 | PointNav fail |
| 27 | tv | 47 | (6.45, -10.70, -2.01) | 5.40 / .150 | 5.109 | 5.40/5.40/5.40 | 17/14 | PointNav fail, no motion |
| 30 | bed | 6 | (-7.73, -2.27, 3.91) | 2.15 / .151 | 1.376 | 1.67/1.67/1.70 | 16/26 | PointNav fail |
| 31 | tv | 37 | (4.78, -7.45, .88) | 11.50 / .029 | 9.505 | 3.22/3.16/3.84 | 16/24 | PointNav fail |

The two rescue rows include one Stable-v0 rescue outside the eight-case core
(12) and one core accepted-no-STOP rescue (25). Consequently:

- all correct-candidate OF-base failures: Oracle A `2/6 = 33.3%`;
- correct-candidate accepted-no-STOP failures: Oracle A `1/5 = 20%`;
- all eight accepted-no-STOP failures: Oracle A `1/8 = 12.5%`.

The four failures show the same pursuit signature as the earlier Stable
Approach audit: 45--62% turn actions, exactly 16--17 attempted FORWARD actions,
little or no net movement, and `rho` that does not fall. Probe 27 moves 0 m;
Probe 30 moves only .04 m; Probe 26 moves .35 m; Probe 31 moves .85 m but ends
farther from its requested endpoint than its minimum.

## Oracle B: direct GT success-viewpoint executor control

Oracle B was generated mechanically from the four correct-candidate A failures.
It did not include any wrong/ambiguous candidate or A success. `Path` is the
acceptance-pose-to-nearest-GT-viewpoint navmesh distance.

| Probe | Target | GT endpoint XYZ | Path (m) | Final GT (m) | rho start/min/final | Net motion (m) | F/T | Attribution |
|---:|---|---|---:|---:|---|---:|---:|---|
| 26 | bed | (-6.95, -2.43, 3.97) | 1.376 | 1.163 | 1.20/1.16/1.16 | .37 | 16/20 | D |
| 27 | tv | (6.11, -10.20, -2.01) | 5.109 | 5.109 | 5.11/5.11/5.11 | 0 | 17/14 | D |
| 30 | bed | (-6.95, -2.43, 3.91) | 1.365 | 1.365 | .66/.66/.66 | 0 | 16/26 | D |
| 31 | tv | (4.63, -5.77, .88) | 9.841 | 9.918 | 2.88/2.81/2.81 | .08 | 16/30 | D |

Oracle B rescue is **0/4**. All four terminate as
`oracle_pointnav_failed_before_endpoint`, with no data/navmesh exception. Probe
30 additionally exposes an existing PointNav detail: its requested GT endpoint
is fixed, but `PointnavPlanner.get_closest_navigable_point` internally shifts
the effective XY goal by .50 m when the Euclidean target is within 1.5 m. This
is part of the unmodified executor, not Oracle endpoint drift. It does not
explain the overall result: the other three B endpoints have zero effective XY
drift and still fail.

The large difference between navmesh path distance and Euclidean `rho` in the
long tv cases is also material. Probe 31 has a 9.84 m legal navmesh path but a
2.88 m PointNav point-goal vector; the learned local executor receives no
explicit global navmesh path and makes only .08 m net progress. Probe 27 makes
no progress at all despite a valid 5.11 m path.

## Required unique taxonomy

| Class | Meaning | Count | Episodes |
|---|---|---:|---|
| A | wrong / ambiguous target candidate | 5 | 14, 19, 24, 28, 29 |
| B | candidate correct, Oracle A succeeds | 2 | 12, 25 |
| C | Oracle A fails, Oracle B succeeds | 0 | — |
| D | GT-viewpoint Oracle B still fails | 4 | 26, 27, 30, 31 |
| E | data, replay, or navmesh anomaly | 0 | — |

This taxonomy covers the eleven OF-base failures in the diagnostic set exactly
once. Probes 32 and 35 are successful OF-base controls, so they are not assigned
a failure class; both remain inside the official success region under A.

## Protection result

| Metric | Result |
|---|---:|
| Episodes / successes | 10 / 10 |
| SR@1m | 100% |
| SPL@1m | .4633 |
| Success losses | 0 |
| Exact success / reason / steps / SPL | 10 / 10 each |
| Exact first accepted event | 10 / 10 |
| Exceptions | 0 |

Protection navigation steps are exactly `30, 29, 27, 231, 381, 444, 488,
453, 39, 410`, matching the frozen full-1000 source episodes. Oracle code is
not instantiated in this replay.

## Artifacts

The versioned machine-readable summary is:

- `results/target_approach_oracle_ceiling_seed20260727/oracle_summary_v1.json`

Candidate artifacts are complete: Oracle A has 13/13 overlays and masks plus
eight correct-candidate pursuit videos; Oracle B has 4/4 overlays, masks, and
pursuit videos. Useful manual-review anchors include:

- Probe 25 rescue: `oracle_a/episode_logs/004_ziup5kvtCCR.basis_0/oracle_assets/`;
- Probe 26 A/B executor failure: `oracle_a/episode_logs/005_qyAac8rV8Zk.basis_25/oracle_assets/`
  and `oracle_b/episode_logs/000_qyAac8rV8Zk.basis_25/oracle_assets/`;
- Probe 27 no-motion A/B failure: `oracle_a/episode_logs/006_zt1RVoi7PcG.basis_17/oracle_assets/`
  and `oracle_b/episode_logs/001_zt1RVoi7PcG.basis_17/oracle_assets/`;
- Probe 19 wrong-candidate overlay: `oracle_a/episode_logs/002_TEEsavR23oF.basis_26/oracle_assets/candidate_gt_overlay.png`;
- Probe 14 boundary-overlap overlay: `oracle_a/episode_logs/001_HY1NcmCgn3n.basis_3/oracle_assets/candidate_gt_overlay.png`.

All paths above are relative to
`results/target_approach_oracle_ceiling_seed20260727/`.

Manual review of the boundary anchors agrees with the numerical rubric. Probe
25's small candidate mask lies on a GT plant surface and supports the correct
classification despite low recall. Probe 19's accepted mask covers a sofa while
GT chairs appear elsewhere in the same frame, supporting a clear association
error. Probe 14's candidate has only a small overlap with the GT plant and is
appropriately retained as ambiguous rather than forced into correct/wrong.

## Final research decision

Do not continue with `Surface-Aware Stagnation Recovery`, additional endpoint
rings/radii, centroid refinement, or success-region fitting. Oracle A already
had endpoints within .152 m geodesic of legal success viewpoints, yet only one
of five correct-candidate accepted-no-STOP failures was rescued. Direct GT
endpoints did not rescue any of the four remaining cases.

The target grounding / approach patch line should therefore be closed. If
target closure is revisited at all, it should be framed as a separate PointNav
executor/global-path integration problem, not another SAM surface or approach
viewpoint heuristic. This conclusion does not argue for modifying that executor
now; it only identifies why further target-grounding patches have a low ceiling.
