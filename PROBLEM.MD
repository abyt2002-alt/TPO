# PROBLEM LOG (Pre-Presentation Validation)

## Scope of this file
This file tracks issues, inconsistencies, risks, and open checks identified during final validation of the tool before presentation.

## How this file will be used
- Record only validated or suspected problems.
- Keep each point concise and testable.
- Do not include fixes here unless explicitly asked later.
- Use this as the single source of truth for what must be reviewed/corrected after discussion.

## Current phase
Discussion and validation only (no code fixes).

## Issue format
For each issue, we will capture:
1. Area (Step/Page/Feature)
2. Problem statement
3. Why it matters
4. Repro steps
5. Expected vs actual
6. Priority (High/Medium/Low)
7. Status (Open/Validated/Deferred)

---

## Logged issues
### Issue 1
1. Area (Step/Page/Feature)  
Step 3 -> Modeling & ROI -> Pack/Combined ROI cards section.
2. Problem statement  
UI labels and grouping are confusing: users are not clear what "Pack Level" vs "Combined" means, and how those cards relate to the slab-level ROI section.
3. Why it matters  
Creates interpretation risk during business review; users may read wrong ROI level and make wrong decisions.
4. Repro steps  
Open Step 3, switch pack tabs, observe top cards ("Combined Topline ROI", "Combined Gross Margin ROI") and slab ROI cards below.
5. Expected vs actual  
Expected: clear, self-explanatory hierarchy (overall combined vs selected pack vs selected slab).  
Actual: terminology/layout causes ambiguity for users.
6. Priority (High/Medium/Low)  
Medium
7. Status (Open/Validated/Deferred)  
Open

### Issue 2
1. Area (Step/Page/Feature)  
Step 3 -> Modeling & ROI -> ROI computation scope.
2. Problem statement  
Step 3 ROI is computed at slab/pack/combined levels without cross-size adjustment effects (12-ML <-> 18-ML interaction), while cross-size behavior is handled later in planner flow.
3. Why it matters  
Potential mismatch between Step 3 ROI interpretation and downstream cross-size-adjusted scenario outputs; stakeholders may assume cross effects are already included when they are not.
4. Repro steps  
Run Step 3 modeling and inspect ROI cards/summaries; compare with Step 4 behavior where cross-size effects are applied.
5. Expected vs actual  
Expected: either cross-size-adjusted ROI in Step 3 or explicit communication that Step 3 ROI excludes cross-size adjustment.  
Actual: Step 3 ROI shown without cross-size adjustment.
6. Priority (High/Medium/Low)  
High
7. Status (Open/Validated/Deferred)  
Open

### Issue 3
1. Area (Step/Page/Feature)  
Step 5 -> Scenario Comparison -> `Max CTS %` filter.
2. Problem statement  
`Max CTS %` filter was not applying correctly because CTS was computed using the wrong revenue field in frontend summary parsing.
3. Why it matters  
Filtering by CTS could return incorrect/no rows, which can mislead scenario selection.
4. Repro steps  
Open Step 5, set `Max CTS %` filter, observe inconsistent filtering behavior (rows not filtered as expected).
5. Expected vs actual  
Expected: CTS filter should consistently filter rows by `scenario_investment / scenario_revenue * 100`.  
Actual (before patch): CTS could become invalid (`NaN`) due to wrong field mapping and filter behavior broke.
6. Priority (High/Medium/Low)  
High
7. Status (Open/Validated/Deferred)  
Validated (fixed in frontend by using `scenario_revenue` when computing CTS).
