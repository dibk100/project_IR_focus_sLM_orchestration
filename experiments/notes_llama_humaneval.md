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


proposed_v1


proposed_v2