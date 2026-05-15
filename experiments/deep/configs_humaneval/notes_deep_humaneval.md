model : "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
dataset : humaneval

single
============================================================
📊 결과 요약
  총 문제: 164
  성공: 3
  실행 성공: 6
  success@1: 0.0183
  execution_success_rate: 0.0366
  conditional_success: 0.5000
============================================================
  code_failed: 158
  define_test_failed: 0
  run_test_failed: 3
============================================================
💾 결과 저장: results/deepcoder7b/humaneval/single/step_logs.jsonl (164건)
💾 결과 저장: results/deepcoder7b/humaneval/single/trajectory_logs.jsonl (164건)
💾 결과 저장: results/deepcoder7b/humaneval/single/summary.json
💾 결과 저장: results/deepcoder7b/humaneval/single/analysis.json

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
💾 결과 저장: results/deepcoder7b/humaneval/repair/step_logs.jsonl (3223건)
💾 결과 저장: results/deepcoder7b/humaneval/repair/trajectory_logs.jsonl (164건)
💾 결과 저장: results/deepcoder7b/humaneval/repair/summary.json
💾 결과 저장: results/deepcoder7b/humaneval/repair/failure_examples.json
📝 failure_examples: 2개 유형 저장됨

planning
============================================================
📊 결과 요약
  총 문제: 164
  성공: 124
  실행 성공: 142
  success@20: 0.7561
  execution_success_rate: 0.8659
  conditional_success: 0.8732
  AUSC: 0.5726
============================================================
  code_failed: 0
  define_test_failed: 0
  run_test_failed: 0
============================================================
  plan 사용: 163/164 (99.4%)
  plan 복구 성공: 123/163 (75.5%)
============================================================
💾 결과 저장: results/deepcoder7b/humaneval/code_then_plan/step_logs.jsonl (1486건)
💾 결과 저장: results/deepcoder7b/humaneval/code_then_plan/trajectory_logs.jsonl (164건)
💾 결과 저장: results/deepcoder7b/humaneval/code_then_plan/summary.json
💾 결과 저장: results/deepcoder7b/humaneval/code_then_plan/failure_examples.json
📝 failure_examples: 13개 유형 저장됨


planning+repair
============================================================
📊 결과 요약
  총 문제: 164
  성공: 119
  실행 성공: 119
  success@20: 0.7256
  execution_success_rate: 0.7256
  conditional_success: 1.0000
  AUSC: 0.5604
============================================================
  code_failed: 0
  define_test_failed: 0
  run_test_failed: 0
============================================================
  plan 사용: 161/164 (98.2%)
  repair 사용: 99/164 (60.4%)
  planning 복구 성공: 116/161 (72.0%)
  repair 복구 성공: 4/99 (4.0%)
============================================================
  [planning-cycle] 사용: 488 cycles (976 calls), 성공: 112/488 (23.0%)
  [repair-call] 사용: 376 calls, 성공: 4/376 (1.1%)
  [call-level] plan→repair 성공: 4/376
💾 결과 저장: results/deepcoder7b/humaneval/code_then_plan_repair/step_logs.jsonl (1516건)
💾 결과 저장: results/deepcoder7b/humaneval/code_then_plan_repair/trajectory_logs.jsonl (164건)
💾 결과 저장: results/deepcoder7b/humaneval/code_then_plan_repair/summary.json
💾 결과 저장: results/deepcoder7b/humaneval/code_then_plan_repair/failure_examples.json
📝 failure_examples: 9개 유형 저장됨


proposed
============================================================
📊 결과 요약
  총 문제: 164
  성공: 120
  실행 성공: 138
  success@20: 0.7317
  execution_success_rate: 0.8415
  conditional_success: 0.8696
  AUSC: 0.5451
============================================================
  code_failed: 26
  define_test_failed: 0
  run_test_failed: 18
============================================================
  [problem-level] plan 사용: 144/164 (87.8%)
  [problem-level] plan 복구 성공: 96/144 (66.7%)
  [problem-level] repair 사용: 159/164 (97.0%)
  [problem-level] repair 복구 성공: 19/159 (11.9%)
  [call-level] planning-cycle 사용: 416 cycles (832 calls), 성공: 96/416 (23.1%)
  [call-level] repair 호출: 446, 성공: 19/446 (4.3%)
============================================================
💾 결과 저장: results/deepcoder7b/humaneval/proposed_v1/summary.json
💾 결과 저장: results/deepcoder7b/humaneval/proposed_v1/step_logs.jsonl (1442건)
💾 결과 저장: results/deepcoder7b/humaneval/proposed_v1/trajectory_logs.jsonl (164건)
✅ policy_loop 완료