phi
============================================================
📊 결과 요약
  총 문제: 164
  성공: 6
  실행 성공: 11
  success@1: 0.0366
  execution_success_rate: 0.0671
  conditional_success: 0.5455
============================================================
  code_failed: 153
  define_test_failed: 0
  run_test_failed: 5
============================================================
💾 결과 저장: results/phi35mini/humaneval/single/step_logs.jsonl (164건)
💾 결과 저장: results/phi35mini/humaneval/single/trajectory_logs.jsonl (164건)
💾 결과 저장: results/phi35mini/humaneval/single/summary.json
💾 결과 저장: results/phi35mini/humaneval/single/analysis.json

repair
============================================================
📊 결과 요약
  총 문제: 164
  성공: 94
  실행 성공: 155
  success@20: 0.5732
  execution_success_rate: 0.9451
  conditional_success: 0.6065
============================================================
  code_failed: 9
  define_test_failed: 0
  run_test_failed: 61
============================================================
💾 결과 저장: results/phi35mini/humaneval/repair/step_logs.jsonl (1591건)
💾 결과 저장: results/phi35mini/humaneval/repair/trajectory_logs.jsonl (164건)
💾 결과 저장: results/phi35mini/humaneval/repair/summary.json
💾 결과 저장: results/phi35mini/humaneval/repair/failure_examples.json
📝 failure_examples: 11개 유형 저장됨

plan에서 걸림
-----------------------------

model : Qwen
dataset : humaneval

Single
============================================================
📊 결과 요약
  총 문제: 164
  성공: 116
  실행 성공: 145
  success@1: 0.7073
  execution_success_rate: 0.8841
  conditional_success: 0.8000
============================================================
  code_failed: 19
  define_test_failed: 0
  run_test_failed: 29
============================================================
💾 결과 저장: results/qwen25coder7b/humaneval/single/step_logs.jsonl (164건)
💾 결과 저장: results/qwen25coder7b/humaneval/single/trajectory_logs.jsonl (164건)
💾 결과 저장: results/qwen25coder7b/humaneval/single/summary.json
💾 결과 저장: results/qwen25coder7b/humaneval/single/analysis.json

Repair
============================================================
📊 결과 요약
  총 문제: 164
  성공: 138
  실행 성공: 144
  success@20: 0.8415
  execution_success_rate: 0.8780
  conditional_success: 0.9583
============================================================
  code_failed: 20
  define_test_failed: 0
  run_test_failed: 6
============================================================
💾 결과 저장: results/qwen25coder7b/humaneval/repair/step_logs.jsonl (718건)
💾 결과 저장: results/qwen25coder7b/humaneval/repair/trajectory_logs.jsonl (164건)
💾 결과 저장: results/qwen25coder7b/humaneval/repair/summary.json
💾 결과 저장: results/qwen25coder7b/humaneval/repair/failure_examples.json
📝 failure_examples: 10개 유형 저장됨

Planning
============================================================
📊 결과 요약
  총 문제: 164
  성공: 152
  실행 성공: 163
  success@20: 0.9268
  execution_success_rate: 0.9939
  conditional_success: 0.9325
  AUSC: 0.8927
============================================================
  code_failed: 0
  define_test_failed: 0
  run_test_failed: 0
============================================================
  plan 사용: 48/164 (29.3%)
  plan 복구 성공: 36/48 (75.0%)
============================================================
💾 결과 저장: results/qwen25coder7b/humaneval/code_then_plan/step_logs.jsonl (492건)
💾 결과 저장: results/qwen25coder7b/humaneval/code_then_plan/trajectory_logs.jsonl (164건)
💾 결과 저장: results/qwen25coder7b/humaneval/code_then_plan/summary.json
💾 결과 저장: results/qwen25coder7b/humaneval/code_then_plan/failure_examples.json
📝 failure_examples: 9개 유형 저장됨

Planning+Repair
============================================================
📊 결과 요약
  총 문제: 164
  성공: 151
  실행 성공: 160
  success@20: 0.9207
  execution_success_rate: 0.9756
  conditional_success: 0.9437
  AUSC: 0.8936
============================================================
  code_failed: 0
  define_test_failed: 0
  run_test_failed: 0
============================================================
  plan 사용: 49/164 (29.9%)
  repair 사용: 20/164 (12.2%)
  planning 복구 성공: 36/49 (73.5%)
  repair 복구 성공: 2/20 (10.0%)
============================================================
  [planning-cycle] 사용: 119 cycles (238 calls), 성공: 34/119 (28.6%)
  [repair-call] 사용: 85 calls, 성공: 2/85 (2.4%)
  [call-level] plan→repair 성공: 2/85
💾 결과 저장: results/qwen25coder7b/humaneval/code_then_plan_repair/step_logs.jsonl (487건)
💾 결과 저장: results/qwen25coder7b/humaneval/code_then_plan_repair/trajectory_logs.jsonl (164건)
💾 결과 저장: results/qwen25coder7b/humaneval/code_then_plan_repair/summary.json
💾 결과 저장: results/qwen25coder7b/humaneval/code_then_plan_repair/failure_examples.json
📝 failure_examples: 11개 유형 저장됨