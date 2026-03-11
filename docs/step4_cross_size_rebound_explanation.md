# Step 4: Why 18-ML Can Increase After Only Changing 12-ML

## What you changed

Example scenario:

- Only one input is changed.
- `12-ML`
- `Dec`
- `slab2`
- Discount moved from `21%` to `24%`

No `18-ML` slab discount is edited directly.

## Why the result still changes for 18-ML

This happens because Step 4 does not keep `18-ML` fixed when only `12-ML` is edited.

The current Step 4 engine does this month by month:

1. Compute `pre_cross` quantity for each size from:
   - projected slab baseline
   - own discount beta
   - lag discount beta
   - other-slab weighted beta
2. Convert each size into a month-level `% change` versus the **previous final month**.
3. Apply cross-size elasticity between `12-ML` and `18-ML`.
4. Produce final month quantities.
5. Use those final month quantities as the previous month input for the next forecast month.

So the correction is chained across months.

## The exact reason the rebound happens

If `12-ML` is increased in `Dec`, then:

1. `12-ML` `Dec` volume rises.
2. Because cross elasticity `18 wrt 12` is negative, `18-ML` gets downward pressure in `Dec`.
3. That lower `18-ML` `Dec final_qty` is then stored and reused as the `prev18` input for `Jan`.
4. In `Jan`, the planner compares unchanged `18-ML pre_cross_qty` against this now-lower `prev18`.
5. That can create a larger positive `% change` for `18-ML` in `Jan`.
6. The same effect can continue into `Feb`.

So even though the first effect on `18-ML` is negative, the chained month-to-month rebasing can create a rebound in later months.

## Current code path

This behavior comes from [step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI app/backend/services/steps/step4_cross_size_planner.py):

- `own12` and `own18` are computed versus previous final month:
  - [step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI app/backend/services/steps/step4_cross_size_planner.py):1471
  - [step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI app/backend/services/steps/step4_cross_size_planner.py):1472
- Cross-size adjusted month result:
  - [step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI app/backend/services/steps/step4_cross_size_planner.py):1478
  - [step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI app/backend/services/steps/step4_cross_size_planner.py):1479
- Final quantities stored back as next month base:
  - [step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI app/backend/services/steps/step4_cross_size_planner.py):1538
  - [step4_cross_size_planner.py](C:/Users/abqua/Desktop/HRI app/backend/services/steps/step4_cross_size_planner.py):1539

## Simple numeric intuition

Assume:

- `18-ML Dec pre_cross = 100`
- Cross effect from `12-ML` pushes `18-ML Dec final` down to `95`

Now for `Jan`:

- `18-ML Jan pre_cross` is still around `100`
- But the planner compares against `prev18 = 95`
- So `Jan own18 change` looks positive:
  - `(100 - 95) / 95 = +5.26%`

That positive rebound can partly or fully offset the earlier negative hit.

If the planner sums `Dec + Jan + Feb`, then the 3-month `18-ML` total can end up higher even though only `12-ML` was changed.

## What this means

This is not random frontend behavior.

It is a direct consequence of the current Step 4 backend design:

- cross effect is applied monthly
- month `t` final becomes month `t+1` base
- this can create rebound behavior

## Why this may feel wrong from a business perspective

Business expectation is usually:

- if `12-ML` becomes more aggressive,
- `18-ML` should get negative pressure,
- and total `18-ML` should usually not rise unless its own inputs are also improved

The current engine can violate that intuition because it mixes:

- same-month cross-pressure
- and sequential rebasing across months

That makes the later months depend on the earlier corrected quantity, not just on the direct scenario for that month.

## Short conclusion

`18-ML` increases after only changing `12-ML` because Step 4 currently:

- applies a negative cross hit in one month,
- then uses that lower final month as the comparison base for the next month,
- which can mechanically create a rebound in later months,
- and the 3-month total can therefore rise.
