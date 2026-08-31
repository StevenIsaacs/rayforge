# Upstream prompt: GlueScript `_current_x`/`_current_y` staleness across jobs

**Target**: ruida-pa maintainers (`external/ruida-pa`, `ruidadriver/rd_gluescript.py`)

## Problem

`GlueScript` tracks the current head position in `_current_x`/`_current_y`
to select the NEAR/FAR move form and to compute deltas. Neither
`new_gluescript()` (rd_gluescript.py:435-464) nor the re-staging reset in
`stage_gluescript(gluescript=...)` (rd_gluescript.py:1699-1717) clears
these fields.

Consequence: when a live backend replays a fresh job's transcript via
`stage_gluescript(transcript)` + `run_job()`, the first `move_xy_to` /
`cut_xy_to` of the new job computes its delta from the *previous job's*
last position. If the previous job ended far from the new job's start
(e.g. a job that ended at (195, 195) followed by a job starting at
(5, 5)), the delta exceeds the 8.192 mm NEAR threshold and the first move
is emitted in the FAR form even though it is a small move. The same
staleness affects the bounding-box expansion and any delta-based logic.

## Reproduction

```python
from ruidadriver.rd_gluescript import GlueScript

gs = GlueScript()
# Job 1 ends far from the origin.
gs.stage_gluescript([
    "declare_job('Job 1', 'MACHINE', [0.0, 0.0], 1, 1, 0.0, 0.0)",
    "move_xy_to(195.0, 195.0)",
    "end_job()",
])
# Job 2 starts near the origin — the first move should be NEAR.
gs.stage_gluescript([
    "declare_job('Job 2', 'MACHINE', [0.0, 0.0], 1, 1, 0.0, 0.0)",
    "move_xy_to(5.0, 5.0)",
    "end_job()",
])
# Expected: MOVE_NEAR_XY nearX=5.000mm nearY=5.000mm
# Actual:   MOVE_FAR_XY X=5.000mm Y=5.000mm (delta computed from 195, 195)
```

## Suggested fix

Reset `_current_x`/`_current_y` (and `_current_z`/`_current_u` if
tracked) to `0.0` in both `new_gluescript()` and the re-staging reset
block of `stage_gluescript()`. A job's first move is always relative to
the job reference point, so the current position should start at the
origin for every new job.

## Context

The rayforge ruidarpa driver treats the GlueScript transcript as the
source of truth and replays it into the live backend for every job
(`stage_gluescript(transcript)` + `run_job()`). This makes the staleness
observable on every job boundary, not just after a session restart.