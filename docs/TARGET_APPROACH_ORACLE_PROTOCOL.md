# Target Approach Oracle Ceiling protocol

Date: 2026-08-28  
Frozen baseline: `of-base-full-v1-v2-20260727` (`3a79d975`)

This branch is an analysis experiment, not a deployable navigation method. It
leaves OF-base target proposal, global Qwen acceptance, frontier selection,
mapping, STOP thresholds, and PointNav controller weights/code unchanged until
the original target acceptance event.

## Frozen sample

`config/evaluation/hm3dv2_approach_oracle_probe23.json` was fixed before any
Oracle result was observed. It contains 13 unique diagnostic episodes:

- all eight Probe 24--31 accepted-no-explicit-STOP failures;
- Stable Approach v0 rescues 12, 14, 19, 25, and 27;
- Stable Approach v0 regression losses 32 and 35.

Indices 25 and 27 belong to both the first and second lists. Ten additional
OF-base successes form the protection replay. They cover the same six target
categories, include short and long target-pursuit trajectories, use distinct
scenes within each category pair, and exclude the 13 diagnostic episodes.

## Candidate audit rubric

The accepted track's latest SAM3 evidence is retained only by the Oracle agent.
The mask is compared with the evaluation-only Habitat semantic IDs in its
source frame. The thresholds are frozen:

- correct: at least 20 overlapping pixels and mask precision at least 0.25;
- wrong: fewer than 10 overlapping pixels or precision below 0.05 when GT is
  visible;
- ambiguous: the middle band, a mask smaller than 20 pixels, absent GT in the
  source frame, missing evidence, or incompatible mask shapes.

Every accepted candidate receives an RGB/mask/GT overlay. Ambiguous cases are
not counted as evidence that a better approach point could recover the target.

## Oracle A

Only a correct candidate proceeds. Its latest mask, depth, pose, and camera
intrinsics are back-projected to a visible surface without GT geometry. Fixed
rings at 0.4, 0.7, 1.0, 1.4, and 2.0 m, sampled at 24 headings each, are snapped
to the current navmesh. Candidates are rejected if snap displacement exceeds
0.5 m, surface clearance lies outside 0.35--2.1 m, the point is unreachable
from the acceptance pose, or a physics ray establishes occlusion. The pose
faces the candidate surface.

GT goal viewpoints rank this already candidate-derived set by minimum navmesh
geodesic distance to a legal success viewpoint. Current-pose path distance,
ring radius, and angle index are deterministic tie-breakers. The selected pose
is fixed for the pursuit and executed by the unmodified OF-base PointNav.

## Oracle B and execution budget

Oracle B is generated mechanically after Oracle A and contains only correct-
candidate Oracle-A failures. It fixes the closest reachable dataset success
viewpoint as the PointNav endpoint. This separates viewpoint-generation limits
from executor limits and is not a deployable policy.

Both Oracles receive the same 300-step post-acceptance allowance. The Habitat
horizon is therefore 800 steps: the official 500-step pre-acceptance horizon
plus a uniform ceiling allowance. An endpoint is considered executed when
PointNav finishes within 0.35 m; GT success is never polled as an online STOP
condition. The normal-success protection group remains at the official 500
steps and uses the ordinary `PointnavAgent`.

## Decision

The result is `GO` only if Oracle A rescues at least half of the correct-
candidate accepted-no-STOP failures and the ten ordinary OF-base protection
replays lose no successes. Otherwise the target grounding/approach direction
is terminated (`NO-GO`). Oracle B success alone cannot change that decision.
