# OF-base Target Closure v0

This branch isolates target closing behavior from the paused geometric-frontier
experiments.  FrontierNet, visual-frontier selection, Qwen frontier utility,
SAM3 proposal generation, PointNav, and the HM3Dv2 1 m protocol are unchanged.

## Implemented closure

The original whole-composition Qwen check remains the recall-oriented pursuit
gate.  Each accepted pursuit also retains its exact SAM mask, source RGB/depth
and camera pose, robust visible surface, candidate-marked evidence, and a
persistent target identity.  Candidate-bound Qwen scores are diagnostic/soft at
initial lock: an uncertain candidate is allowed one better view instead of being
permanently rejected.  A fresh candidate-bound score is mandatory for STOP.

Mask-edge pixels are eroded when sufficient area remains, depth outliers are
removed with a median/MAD rule, and the current visible surface is represented
by a bounded world-point sample and coordinate median.  Cross-observation
object merging retains the latest strong bound geometry rather than averaging
centroids indefinitely.

Approach candidates are sampled at 0.7 m from the visible surface along the
original viewing direction and four small angular alternatives.  Habitat's
navmesh supplies snap, displacement rejection, reachability, and geodesic
ranking; the existing learned PointNav controller executes the selected pose.
Each pose faces the bound surface.

`path_exhausted` now means that one approach waypoint was consumed.  It enters
a bounded re-observation cycle while retaining target intent.  STOP requires:

1. an active hypothesis;
2. a freshly re-associated SAM candidate;
3. candidate-bound Qwen probability at least 0.7;
4. an exhausted/completed approach;
5. horizontal distance to the visible surface below 1 m; and
6. camera heading within 60 degrees of the surface.

Two unsuccessful re-observation cycles or 120 pursuit steps release the target.
No Habitat goal geometry or semantic GT is read by policy.  Semantic-mask
overlap and final evaluator distance are logged strictly for later attribution.

## Paired ProbeSet

`config/evaluation/hm3dv2_target_closure_probe64.json` is frozen from the
OF-base HM3Dv2 full-1000 run.  Its source manifest SHA-256 is
`c7ef8f4bcc42a54d29932c71ff6371e46bccb8e720ad9afaa9f89df2e6271374`.
The first 32 entries are target-closure failures (false commit without visible
GT, false commit with visible GT, and accepted-without-STOP); the last 32 are
matched OF-base successes covering far/near-centroid `path_exhausted` and normal
distance STOP.  Every entry records its exact full-run `source_index`.

Run or resume it with:

```bash
bash scripts/run_target_closure_probe64.sh
```

The acceptance contract is at least 6/32 failure rescues, no more than 3/32
regression losses, positive paired net gain, and no more than 15% increase in
median common-success steps.  These criteria were fixed before the run.
