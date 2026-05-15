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
============================================================
📊 결과 요약
  총 문제: 257
  성공: 204
  실행 성공: 222
  success@20: 0.7938
  execution_success_rate: 0.8638
  conditional_success: 0.9189
  AUSC: 0.6840
============================================================
  code_failed: 0
  define_test_failed: 0
  run_test_failed: 0
============================================================
  plan 사용: 242/257 (94.2%)
  plan 복구 성공: 189/242 (78.1%)
============================================================
💾 결과 저장: results/deepcoder7b/mbpp/code_then_plan/step_logs.jsonl (1775건)
💾 결과 저장: results/deepcoder7b/mbpp/code_then_plan/trajectory_logs.jsonl (257건)
💾 결과 저장: results/deepcoder7b/mbpp/code_then_plan/summary.json
💾 결과 저장: results/deepcoder7b/mbpp/code_then_plan/failure_examples.json
📝 failure_examples: 15개 유형 저장됨

planning_repair
============================================================
📊 결과 요약
  총 문제: 257
  성공: 211
  실행 성공: 229
  success@20: 0.8210
  execution_success_rate: 0.8911
  conditional_success: 0.9214
  AUSC: 0.7107
============================================================
  code_failed: 0
  define_test_failed: 0
  run_test_failed: 0
============================================================
  plan 사용: 242/257 (94.2%)
  repair 사용: 87/257 (33.9%)
  planning 복구 성공: 196/242 (81.0%)
  repair 복구 성공: 25/87 (28.7%)
============================================================
  [planning-cycle] 사용: 522 cycles (1044 calls), 성공: 171/522 (32.8%)
  [repair-call] 사용: 351 calls, 성공: 25/351 (7.1%)
  [call-level] plan→repair 성공: 25/351
💾 결과 저장: results/deepcoder7b/mbpp/code_then_plan_repair/step_logs.jsonl (1652건)
💾 결과 저장: results/deepcoder7b/mbpp/code_then_plan_repair/trajectory_logs.jsonl (257건)
💾 결과 저장: results/deepcoder7b/mbpp/code_then_plan_repair/summary.json
💾 결과 저장: results/deepcoder7b/mbpp/code_then_plan_repair/failure_examples.json
📝 failure_examples: 14개 유형 저장됨

============================================================
📊 결과 요약
  총 문제: 257
  성공: 212
  실행 성공: 227
  success@20: 0.8249
  execution_success_rate: 0.8833
  conditional_success: 0.9339
  AUSC: 0.7574
============================================================
  code_failed: 30
  define_test_failed: 0
  run_test_failed: 15
============================================================
  [problem-level] plan 사용: 73/257 (28.4%)
  [problem-level] plan 복구 성공: 23/73 (31.5%)
  [problem-level] repair 사용: 242/257 (94.2%)
  [problem-level] repair 복구 성공: 174/242 (71.9%)
  [call-level] planning-cycle 사용: 263 cycles (526 calls), 성공: 23/263 (8.7%)
  [call-level] repair 호출: 555, 성공: 174/555 (31.4%)
============================================================
💾 결과 저장: results/deepcoder7b/mbpp/proposed_v1/summary.json
💾 결과 저장: results/deepcoder7b/mbpp/proposed_v1/step_logs.jsonl (1338건)
💾 결과 저장: results/deepcoder7b/mbpp/proposed_v1/trajectory_logs.jsonl (257건)
✅ policy_loop 완료
