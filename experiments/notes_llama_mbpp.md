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
============================================================
📊 결과 요약
  총 문제: 257
  성공: 183
  실행 성공: 224
  success@20: 0.7121
  execution_success_rate: 0.8716
  conditional_success: 0.8170
  AUSC: 0.5883
============================================================
  code_failed: 0
  define_test_failed: 0
  run_test_failed: 0
============================================================
  plan 사용: 231/257 (89.9%)
  plan 복구 성공: 157/231 (68.0%)
============================================================
💾 결과 저장: results/llama318b/mbpp/code_then_plan/step_logs.jsonl (2225건)
💾 결과 저장: results/llama318b/mbpp/code_then_plan/trajectory_logs.jsonl (257건)
💾 결과 저장: results/llama318b/mbpp/code_then_plan/summary.json
💾 결과 저장: results/llama318b/mbpp/code_then_plan/failure_examples.json
📝 failure_examples: 15개 유형 저장됨

planning_repair
============================================================
📊 결과 요약
  총 문제: 257
  성공: 200
  실행 성공: 251
  success@20: 0.7782
  execution_success_rate: 0.9767
  conditional_success: 0.7968
  AUSC: 0.6574
============================================================
  code_failed: 0
  define_test_failed: 0
  run_test_failed: 0
============================================================
  plan 사용: 229/257 (89.1%)
  repair 사용: 142/257 (55.3%)
  planning 복구 성공: 172/229 (75.1%)
  repair 복구 성공: 67/142 (47.2%)
============================================================
  [planning-cycle] 사용: 584 cycles (1168 calls), 성공: 105/584 (18.0%)
  [repair-call] 사용: 479 calls, 성공: 67/479 (14.0%)
  [call-level] plan→repair 성공: 67/479
💾 결과 저장: results/llama318b/mbpp/code_then_plan_repair/step_logs.jsonl (1904건)
💾 결과 저장: results/llama318b/mbpp/code_then_plan_repair/trajectory_logs.jsonl (257건)
💾 결과 저장: results/llama318b/mbpp/code_then_plan_repair/summary.json
💾 결과 저장: results/llama318b/mbpp/code_then_plan_repair/failure_examples.json
📝 failure_examples: 16개 유형 저장됨

proposed_v1
============================================================
📊 결과 요약
  총 문제: 257
  성공: 190
  실행 성공: 249
  success@20: 0.7393
  execution_success_rate: 0.9689
  conditional_success: 0.7631
  AUSC: 0.6206
============================================================
  code_failed: 8
  define_test_failed: 0
  run_test_failed: 59
============================================================
  [problem-level] plan 사용: 183/257 (71.2%)
  [problem-level] plan 복구 성공: 64/183 (35.0%)
  [problem-level] repair 사용: 219/257 (85.2%)
  [problem-level] repair 복구 성공: 96/219 (43.8%)
  [call-level] planning-cycle 사용: 465 cycles (930 calls), 성공: 64/465 (13.8%)
  [call-level] repair 호출: 776, 성공: 96/776 (12.4%)
============================================================
💾 결과 저장: results/llama318b/mbpp/proposed_v1/summary.json
💾 결과 저장: results/llama318b/mbpp/proposed_v1/step_logs.jsonl (1963건)
💾 결과 저장: results/llama318b/mbpp/proposed_v1/trajectory_logs.jsonl (257건)
✅ policy_loop 완료

proposed_v2