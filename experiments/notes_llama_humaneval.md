model : meta-llama/Llama-3.1-8B-Instruct
dataset : humaneval

single
============================================================
📊 결과 요약
  총 문제: 164
  성공: 56
  실행 성공: 89
  success@1: 0.3415
  execution_success_rate: 0.5427
  conditional_success: 0.6292
============================================================
  code_failed: 75
  define_test_failed: 0
  run_test_failed: 33
============================================================
💾 결과 저장: results/llama318b/humaneval/single/step_logs.jsonl (164건)
💾 결과 저장: results/llama318b/humaneval/single/trajectory_logs.jsonl (164건)
💾 결과 저장: results/llama318b/humaneval/single/summary.json

repair
============================================================
📊 결과 요약
  총 문제: 164
  성공: 50
  실행 성공: 50
  success@20: 0.3049
  execution_success_rate: 0.3049
  conditional_success: 1.0000
  AUSC: 0.3049
============================================================
  code_failed: 114
  define_test_failed: 0
  run_test_failed: 0
============================================================
💾 결과 저장: results/llama318b/humaneval/repair/step_logs.jsonl (2330건)
💾 결과 저장: results/llama318b/humaneval/repair/trajectory_logs.jsonl (164건)
💾 결과 저장: results/llama318b/humaneval/repair/summary.json
💾 결과 저장: results/llama318b/humaneval/repair/failure_examples.json
📝 failure_examples: 4개 유형 저장됨


planning

planning_repair
============================================================
📊 결과 요약
  총 문제: 164
  성공: 105
  실행 성공: 115
  success@20: 0.6402
  execution_success_rate: 0.7012
  conditional_success: 0.9130
  AUSC: 0.5369
============================================================
  code_failed: 0
  define_test_failed: 0
  run_test_failed: 0
============================================================
  plan 사용: 105/164 (64.0%)
  repair 사용: 90/164 (54.9%)
  planning 복구 성공: 46/105 (43.8%)
  repair 복구 성공: 7/90 (7.8%)
============================================================
  [planning-cycle] 사용: 480 cycles (960 calls), 성공: 39/480 (8.1%)
  [repair-call] 사용: 441 calls, 성공: 7/441 (1.6%)
  [call-level] plan→repair 성공: 7/441

proposed_v1


proposed_v2