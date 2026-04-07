## Repair Loop Result Summary

### Overall
- Single-shot pass@1: 0.7195 (118/164)
- Repair final success rate: 0.8415 (138/164)
- Absolute improvement: +0.1220
- Repair gain: 20 tasks

### Solved on Attempt
- Attempt 0: 118
- Attempt 1: 11 additional tasks
- Attempt 2: 9 additional tasks

### Key Observations
- Repair loop provides substantial improvement over single-shot baseline.
- A large portion of initially failed tasks can be recovered through iterative repair.
- However, repair is not always stable: some assertion failures are transformed into syntax or name errors during later attempts.
- This suggests that repair is an effective but fragile orchestration primitive.

### Interpretation
- The improvement confirms that orchestration contributes beyond single-call inference.
- At the same time, the instability of later repair attempts indicates that naive iterative repair has limits.
- This motivates testing more structured orchestration strategies (e.g., planner/coder separation).