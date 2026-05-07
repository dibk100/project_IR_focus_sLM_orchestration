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

repair 다시
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
📝 failure_examples: 4개 유형 저장됨


code_then_plan_repair
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