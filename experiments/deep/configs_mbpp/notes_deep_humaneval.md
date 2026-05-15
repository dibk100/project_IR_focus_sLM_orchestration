model : "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
dataset : mbpp

single
============================================================
📊 결과 요약
  총 문제: 257
  성공: 10
  실행 성공: 12
  success@1: 0.0389
  execution_success_rate: 0.0467
  conditional_success: 0.8333
============================================================
  code_failed: 245
  define_test_failed: 0
  run_test_failed: 2
============================================================
💾 결과 저장: results/deepcoder7b/mbpp/single/step_logs.jsonl (257건)
💾 결과 저장: results/deepcoder7b/mbpp/single/trajectory_logs.jsonl (257건)
💾 결과 저장: results/deepcoder7b/mbpp/single/summary.json
💾 결과 저장: results/deepcoder7b/mbpp/single/analysis.json

repair
============================================================
📊 결과 요약
  총 문제: 257
  성공: 152
  실행 성공: 153
  success@20: 0.5914
  execution_success_rate: 0.5953
  conditional_success: 0.9935
  AUSC: 0.4140
============================================================
  code_failed: 104
  define_test_failed: 0
  run_test_failed: 1
============================================================
💾 결과 저장: results/deepcoder7b/mbpp/repair/step_logs.jsonl (3164건)
💾 결과 저장: results/deepcoder7b/mbpp/repair/trajectory_logs.jsonl (257건)
💾 결과 저장: results/deepcoder7b/mbpp/repair/summary.json
💾 결과 저장: results/deepcoder7b/mbpp/repair/failure_examples.json
📝 failure_examples: 8개 유형 저장됨

planning

