model : meta-llama/Llama-3.1-8B-Instruct
dataset : mbpp

single
============================================================
📊 결과 요약
  총 문제: 257
  성공: 23
  실행 성공: 212
  success@1: 0.0895
  execution_success_rate: 0.8249
  conditional_success: 0.1085
============================================================
  code_failed: 45
  define_test_failed: 0
  run_test_failed: 189
============================================================
💾 결과 저장: results/llama318b/mbpp/single/step_logs.jsonl (257건)
💾 결과 저장: results/llama318b/mbpp/single/trajectory_logs.jsonl (257건)
💾 결과 저장: results/llama318b/mbpp/single/summary.json
💾 결과 저장: results/llama318b/mbpp/single/analysis.json

repair
============================================================
📊 결과 요약
  총 문제: 257
  성공: 60
  실행 성공: 212
  success@20: 0.2335
  execution_success_rate: 0.8249
  conditional_success: 0.2830
  AUSC: 0.2103
============================================================
  code_failed: 45
  define_test_failed: 0
  run_test_failed: 152
============================================================
💾 결과 저장: results/llama318b/mbpp/repair/step_logs.jsonl (4119건)
💾 결과 저장: results/llama318b/mbpp/repair/trajectory_logs.jsonl (257건)
💾 결과 저장: results/llama318b/mbpp/repair/summary.json
💾 결과 저장: results/llama318b/mbpp/repair/failure_examples.json
📝 failure_examples: 11개 유형 저장됨


planning

planning_repair


proposed_v1


proposed_v2