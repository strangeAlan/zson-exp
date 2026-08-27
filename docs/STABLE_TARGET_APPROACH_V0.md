# Stable Target Approach v0: pre-development audit and implementation record

Date: 2026-08-27  
Frozen baseline: `of-base-full-v1-v2-20260727` (`6037eb3`)  
Development branch: `research/stable-target-approach`

## Audit gate

The audit covers exactly the eight `failure_accepted_no_stop` episodes at
ProbeSet indices 24--31. The existing evaluator did not save per-step camera
poses or generated goal poses (`write_path=None`), so historic positional goal
drift cannot be reconstructed exactly. The raw navigation log does retain the
per-step PointNav action, `rho`, heat state, object-centroid distance, target
visibility, acceptance pose, raw centroid, and first approach endpoint.

| Probe | Target | End | Post-accept steps | F/L/R | rho start/min/end | Last-50 turn | Longest constant object-distance run | GT visible after accept |
|---:|---|---|---:|---|---|---:|---:|---:|
| 24 | chair | max steps | 295 | 137/85/72 | 1.09/1.08/1.17 | 0.48 | 80 | 45/295 |
| 25 | plant | max steps | 264 | 30/115/118 | 1.94/1.94/2.41 | 0.86 | 11 | 204/264 |
| 26 | bed | robot stuck | 194 | 79/76/38 | 1.77/1.77/1.77 | 0.58 | 194 | 192/194 |
| 27 | tv | robot stuck | 190 | 105/56/28 | 6.26/6.26/6.26 | 0.44 | 190 | 70/190 |
| 28 | toilet | robot stuck | 158 | 69/54/31 | 1.82/1.62/1.62 | 0.54 | 158 | 0/158 |
| 29 | sofa | max steps | 405 | 184/129/91 | 1.53/1.27/1.84 | 0.54 | 123 | 330/405 |
| 30 | bed | max steps | 368 | 156/116/95 | 1.79/1.71/1.85 | 0.54 | 153 | 190/368 |
| 31 | tv | max steps | 25 | 9/8/7 | 2.06/2.02/2.02 | 0.60 | 18 | 15/25 |

All eight have repeated 20-step windows without meaningful `rho` improvement.
Three remain position-constant for their entire 158--194-step pursuit. In the
other five, PointNav still fails to converge and often moves away again.

### Endpoint drift and controller reset

OF-base regenerates the locked-object goal on every navigation cycle in
`FrontierManager.get_goal_pose`: its position is selected along the *current*
camera-to-centroid ray and its rotation is copied from the *current* camera.
Therefore the full goal pose changes after every turn even if its position does
not. `PointnavPlanner.update_start_goal` compares the full 4x4 pose and resets
`minimum_rho`, `close_enough`, forward heat, and rotation heat whenever it
changes. This is directly reflected by the logs: despite hundreds of pursuit
steps, heat usually remains 0 or 1. The implementation thus guarantees
orientation drift/reset and permits positional drift; the old logs do not
contain enough state to quantify the latter exactly.

### Endpoint validity

The OF helper checks Wavemap occupancy along a centroid ray, but does not require
Habitat-navmesh navigability, same-island reachability, clearance, or a
target-facing stop pose. Four of the eight first approach endpoints are exactly
the raw object centroid.

A read-only replay placed each logged first endpoint into its episode's Habitat
navmesh:

| Probe | Same-island path after snap | XY snap displacement (m) |
|---:|---:|---:|
| 24 | no | 0.068 |
| 25 | no | 0.438 |
| 26 | no | 0.132 |
| 27 | yes | 0.788 |
| 28 | yes | 0.815 |
| 29 | yes | 0.254 |
| 30 | no | 0.105 |
| 31 | yes | 1.039 |

Thus 4/8 are disconnected from the acceptance position after snap; three of
the four remaining endpoints need a 0.79--1.04 m correction. The audit supports
both an endpoint-validity defect and a systematic PointNav pursuit/reset defect.
It therefore satisfies the requested development gate.

## External mechanisms used as design references

- [OpenFrontier](https://github.com/cvg/OpenFrontier) remains the behavioral
  baseline; no detector, frontier, utility, or evaluator code is replaced.
- [ApexNav](https://github.com/Robotics-STAR-Lab/ApexNav) validates target-side
  collision-free path points at several safety distances and retains paths to
  reduce oscillation. Only the small reachable-viewpoint idea is used.
- [BeliefMapNav](https://github.com/ZiboKNOW/BeliefMapNav) applies target-goal
  hysteresis and bounded navigation attempts. Only those control principles are
  used; its mapping and policy are not imported.
- [ConsistNav](https://arxiv.org/abs/2605.09869) motivates persistent intent,
  progress/stall monitoring, rotational-stagnation detection, and bounded
  recovery. No unpublished implementation detail is assumed.

## v0 implementation

Before global Qwen acceptance, execution is OF-base. After acceptance only:

1. Generate a small set from the legacy endpoint and target-side radial
   viewpoints; snap each to Habitat navmesh, require finite same-island path,
   reject large snap and points within 0.35 m of the raw centroid, and orient it
   toward the target.
2. Prefer the legacy endpoint if it passes those checks. Otherwise use the
   fixed best reachable viewpoint. Keep its full 4x4 pose unchanged during the
   pursuit cycle.
3. Maintain a 20-step window of `rho`, endpoint distance, actual XY motion, and
   TURN/FORWARD actions. A stall requires both `rho` and endpoint improvement
   below 0.12 m plus high turning or less than 0.15 m net translation.
4. At a stall, retain the object lock, run one SAM re-observation, then fix the
   next reachable viewpoint. Permit at most two endpoint changes before release.
5. Preserve OF-base's `raw centroid < 1 m` STOP. Preserve path-exhausted STOP
   only when the robot is within 0.45 m of the stable endpoint; a far exhausted
   path is treated as a recoverable pursuit failure.

Diagnostics store all generated/rejected endpoints, snap and initial geodesic
distance, per-step `rho` and navmesh distance, actual motion, action ratios,
stagnation triggers, re-observation, recovery, release, and STOP reason.

## Final one-episode smoke

The final smoke used Probe 26 (`qyAac8rV8Zk`, episode 25, bed), whose baseline
pursuit was position-constant for 194 steps. It completed without exception.
The stable endpoint remained fixed; stalls were correctly detected at steps 70,
90, and 110; two distinct navmesh-valid endpoints were tried before bounded
release. The episode was not rescued. This is useful negative evidence: even
after endpoint stability and reachability are repaired, this case shows a
remaining low-level PointNav execution failure near the doorway. The full frozen
ProbeSet is required to decide whether the mechanism has any repeatable rescue
without regressions.
