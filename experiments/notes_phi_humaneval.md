model : microsoft/Phi-3.5-mini-instruct
dataset : humaneval

single
============================================================
📊 결과 요약
  총 문제: 164
  성공: 2
  실행 성공: 6
  success@1: 0.0122
  execution_success_rate: 0.0366
  conditional_success: 0.3333
============================================================
  code_failed: 158
  define_test_failed: 0
  run_test_failed: 4
============================================================
💾 결과 저장: results/phi35mini/humaneval/single/step_logs.jsonl (164건)
💾 결과 저장: results/phi35mini/humaneval/single/trajectory_logs.jsonl (164건)
💾 결과 저장: results/phi35mini/humaneval/single/summary.json
💾 결과 저장: results/phi35mini/humaneval/single/analysis.json

repair
============================================================
📊 결과 요약
  총 문제: 164
  성공: 3
  실행 성공: 3
  success@20: 0.0183
  execution_success_rate: 0.0183
  conditional_success: 1.0000
  AUSC: 0.0183
============================================================
  code_failed: 161
  define_test_failed: 0
  run_test_failed: 0
============================================================
💾 결과 저장: results/phi35mini/humaneval/repair/step_logs.jsonl (3223건)
💾 결과 저장: results/phi35mini/humaneval/repair/trajectory_logs.jsonl (164건)
💾 결과 저장: results/phi35mini/humaneval/repair/summary.json
💾 결과 저장: results/phi35mini/humaneval/repair/failure_examples.json
📝 failure_examples: 2개 유형 저장됨

planning
============================================================
📊 결과 요약
  총 문제: 164
  성공: 118
  실행 성공: 147
  success@20: 0.7195
  execution_success_rate: 0.8963
  conditional_success: 0.8027
  AUSC: 0.6055
============================================================
  code_failed: 0
  define_test_failed: 0
  run_test_failed: 0
============================================================
  plan 사용: 0/0
  plan 복구 성공: 0/0
============================================================
💾 결과 저장: results/phi35mini/humaneval/code_then_plan/failure_examples.json
📝 failure_examples: 8개 유형 저장됨


planning_repair
============================================================
📊 결과 요약
  총 문제: 164
  성공: 117
  실행 성공: 121
  success@20: 0.7134
  execution_success_rate: 0.7378
  conditional_success: 0.9669
  AUSC: 0.5689
============================================================
  code_failed: 0
  define_test_failed: 0
  run_test_failed: 0
============================================================
  plan 사용: 162/164 (98.8%)
  repair 사용: 82/164 (50.0%)
  planning 복구 성공: 115/162 (71.0%)
  repair 복구 성공: 1/82 (1.2%)
============================================================
  [planning-cycle] 사용: 478 cycles (956 calls), 성공: 114/478 (23.8%)
  [repair-call] 사용: 364 calls, 성공: 1/364 (0.3%)
  [call-level] plan→repair 성공: 1/364
💾 결과 저장: results/phi35mini/humaneval/code_then_plan_repair/step_logs.jsonl (1484건)
💾 결과 저장: results/phi35mini/humaneval/code_then_plan_repair/trajectory_logs.jsonl (164건)
💾 결과 저장: results/phi35mini/humaneval/code_then_plan_repair/summary.json
💾 결과 저장: results/phi35mini/humaneval/code_then_plan_repair/failure_examples.json
📝 failure_examples: 10개 유형 저장됨

proposed-v1
============================================================
📊 결과 요약
  총 문제: 164
  성공: 126
  실행 성공: 150
  success@20: 0.7683
  execution_success_rate: 0.9146
  conditional_success: 0.8400
  AUSC: 0.6466
============================================================
  code_failed: 14
  define_test_failed: 0
  run_test_failed: 24
============================================================
  [problem-level] plan 사용: 114/164 (69.5%)
  [problem-level] plan 복구 성공: 76/114 (66.7%)
  [problem-level] repair 사용: 159/164 (97.0%)
  [problem-level] repair 복구 성공: 45/159 (28.3%)
  [call-level] planning-cycle 사용: 289 cycles (578 calls), 성공: 76/289 (26.3%)
  [call-level] repair 호출: 363, 성공: 45/363 (12.4%)
============================================================
💾 결과 저장: results/phi35mini/humaneval/proposed_v1/summary.json
💾 결과 저장: results/phi35mini/humaneval/proposed_v1/step_logs.jsonl (1105건)
💾 결과 저장: results/phi35mini/humaneval/proposed_v1/trajectory_logs.jsonl (164건)
✅ policy_loop 완료

============================================================
📊 결과 요약
  총 문제: 164
  성공: 125
  실행 성공: 152
  success@20: 0.7622
  execution_success_rate: 0.9268
  conditional_success: 0.8224
  AUSC: 0.6122
============================================================
  code_failed: 12
  define_test_failed: 0
  run_test_failed: 27
============================================================
  [problem-level] plan 사용: 117/164 (71.3%)
  [problem-level] plan 복구 성공: 78/117 (66.7%)
  [problem-level] repair 사용: 161/164 (98.2%)
  [problem-level] repair 복구 성공: 44/161 (27.3%)
  [call-level] planning-cycle 사용: 335 cycles (670 calls), 성공: 78/335 (23.3%)
  [call-level] repair 호출: 387, 성공: 44/387 (11.4%)
============================================================
💾 결과 저장: results/phi35mini/humaneval/proposed_v1/summary.json
💾 결과 저장: results/phi35mini/humaneval/proposed_v1/step_logs.jsonl (1221건)
💾 결과 저장: results/phi35mini/humaneval/proposed_v1/trajectory_logs.jsonl (164건)
✅ policy_loop 완료
