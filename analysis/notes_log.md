# Table 1: Success@k recovery dynamics across refinement strategies under a fixed inference budget of 20 model calls on HumanEval.

| k (Call Budget) | Single-shot | Repair | Planning | Planning-Repair |
| --------------- | ----------: | -----: | -------: | --------------: |
| 1               |       70.7% |  68.9% |    70.7% |           70.1% |
| 2               |       70.7% |  74.4% |    70.7% |           70.1% |
| 3               |       70.7% |  80.5% |    89.0% |           87.8% |
| 4               |       70.7% |  81.7% |    89.0% |           89.0% |
| 5               |       70.7% |  83.5% |    89.6% |           89.0% |
| 6               |       70.7% |  83.5% |    89.6% |           92.1% |
| 7               |       70.7% |  83.5% |    91.5% |           92.1% |
| 8               |       70.7% |  83.5% |    91.5% |           92.1% |
| 9               |       70.7% |  83.5% |    91.5% |           92.1% |
| 10              |       70.7% |  83.5% |    91.5% |           92.1% |
| 11              |       70.7% |  83.5% |    91.5% |           92.1% |
| 12              |       70.7% |  83.5% |    91.5% |           92.1% |
| 13              |       70.7% |  83.5% |    92.1% |           92.1% |
| 14              |       70.7% |  84.1% |    92.1% |           92.1% |
| 15              |       70.7% |  84.1% |    92.1% |           92.1% |
| 16              |       70.7% |  84.1% |    92.1% |           92.1% |
| 17              |       70.7% |  84.1% |    92.1% |           92.1% |
| 18              |       70.7% |  84.1% |    92.1% |           92.1% |
| 19              |       70.7% |  84.1% |    92.7% |           92.1% |
| 20              |       70.7% |  84.1% |    92.7% |           92.1% |

# Table X. Dominant refinement trajectory patterns across strategies
Table X summarizes representative refinement trajectory patterns observed across different refinement strategies. Rather than focusing only on final success rates, the table highlights how failures evolve during iterative refinement under bounded inference budgets.

| Strategy        | Representative Transition Pattern                                       | Count | Calls | Outcome | Interpretation                                                                                          |
| --------------- | ----------------------------------------------------------------------- | ----: | ----: | ------- | ------------------------------------------------------------------------------------------------------- |
| Repair          | `EXEC_FAIL:SyntaxError → PASS`                                          |     4 |     2 | Success | Localized syntactic failures are often recoverable through minimal repair.                              |
| Repair          | `EXEC_FAIL:SyntaxError → EXEC_FAIL:SyntaxError → PASS`                  |     8 |     3 | Success | Some executable failures require multiple repair iterations before recovery.                            |
| Repair          | `TEST_FAIL:AssertionError → PASS`                                       |     4 |     2 | Success | Certain semantic failures are recoverable through localized correction.                                 |
| Repair          | `TEST_FAIL:AssertionError → EXEC_FAIL:NameError → ...`                  |     4 |    20 | Failure | Semantic failures frequently collapse into persistent executable failure loops during iterative repair. |
| Repair          | `TEST_FAIL:AssertionError → TEST_FAIL:AssertionError → ...`             |     2 |    20 | Failure | Persistent semantic failures often remain unresolved under bounded budgets.                             |
| Planning        | `EXEC_FAIL:SyntaxError → PLAN_DONE → PASS`                              |    13 |     3 | Success | Explicit planning enables rapid structural recovery from executable failures.                           |
| Planning        | `TEST_FAIL:AssertionError → PLAN_DONE → PASS`                           |    11 |     3 | Success | Planning-guided regeneration improves semantic correction effectiveness.                                |
| Planning        | `EXEC_FAIL:NameError → PLAN_DONE → PASS`                                |     3 |     3 | Success | Planning successfully escapes executable failure states that persist under repair-only refinement.      |
| Planning        | `TEST_FAIL:AssertionError → PLAN_DONE → ...`                            |     6 |    20 | Failure | Certain semantic failures persist despite repeated planning-guided regeneration.                        |
| Planning-Repair | `TEST_FAIL:AssertionError → PLAN_DONE → PASS`                           |    13 |     3 | Success | Sequential refinement preserves strong semantic recovery behavior.                                      |
| Planning-Repair | `EXEC_FAIL:SyntaxError → PLAN_DONE → PASS`                              |    12 |     3 | Success | Planning-guided regeneration remains effective for executable failures.                                 |
| Planning-Repair | `TEST_FAIL:AssertionError → PLAN_DONE → TEST_FAIL:AssertionError → ...` |     3 |    20 | Failure | Sequential refinement still exhibits persistent semantic failure trajectories under bounded budgets.    |
| Planning-Repair | `TEST_FAIL:AssertionError → PLAN_DONE → EXEC_FAIL:NameError → ...`      |     1 |    20 | Failure | Sequential refinement occasionally introduces mixed semantic-executable failure oscillations.           |

# stagnation_summary

| Strategy        | TEST stagnation ≥3 | EXEC stagnation ≥3 | TEST stagnation ≥5 | EXEC stagnation ≥5 | Interpretation                                                                                              |
| --------------- | -----------------: | -----------------: | -----------------: | -----------------: | ----------------------------------------------------------------------------------------------------------- |
| Single-shot     |                  0 |                  0 |                  0 |                  0 | No iterative refinement is performed, so persistent stagnation cannot occur.                                |
| Repair          |                  8 |                 23 |                  7 |                 21 | Repair frequently enters persistent executable-failure loops, particularly repeated EXEC_FAIL trajectories. |
| Planning        |                  0 |                  0 |                  0 |                  0 | Planning-based refinement rarely exhibits persistent stagnation under bounded budgets.                      |
| Planning-Repair |                  0 |                  0 |                  0 |                  0 | Sequential planning and repair also avoid long-horizon stagnation loops.                                    |

# Failure Transition Matrix
repair는 어떤 failure에서 어떤 failure로 변하는가?


