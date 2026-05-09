model : microsoft/Phi-3.5-mini-instruct
dataset : mbpp

single
============================================================
📊 결과 요약
  총 문제: 257
  성공: 163
  실행 성공: 194
  success@1: 0.6342
  execution_success_rate: 0.7549
  conditional_success: 0.8402
============================================================
  code_failed: 63
  define_test_failed: 0
  run_test_failed: 31
============================================================
💾 결과 저장: results/phi/mbpp/single/step_logs.jsonl (257건)
💾 결과 저장: results/phi/mbpp/single/trajectory_logs.jsonl (257건)
💾 결과 저장: results/phi/mbpp/single/summary.json

repair
============================================================
📊 결과 요약
  총 문제: 257
  성공: 179
  실행 성공: 198
  success@20: 0.6965
  execution_success_rate: 0.7704
  conditional_success: 0.9040
  AUSC: 0.6887
============================================================
  code_failed: 59
  define_test_failed: 0
  run_test_failed: 19
============================================================
💾 결과 저장: results/phi/mbpp/repair/step_logs.jsonl (1779건)
💾 결과 저장: results/phi/mbpp/repair/trajectory_logs.jsonl (257건)
💾 결과 저장: results/phi/mbpp/repair/summary.json
💾 결과 저장: results/phi/mbpp/repair/failure_examples.json
📝 failure_examples: 12개 유형 저장됨

planning
============================================================
📊 결과 요약
  총 문제: 257
  성공: 200
  실행 성공: 223
  success@20: 0.7782
  execution_success_rate: 0.8677
  conditional_success: 0.8969
  AUSC: 0.7471
============================================================
  code_failed: 0
  define_test_failed: 0
  run_test_failed: 0
============================================================
  plan 사용: 94/257 (36.6%)
  plan 복구 성공: 37/94 (39.4%)
============================================================
💾 결과 저장: results/phi/mbpp/code_then_plan/step_logs.jsonl (1443건)
💾 결과 저장: results/phi/mbpp/code_then_plan/trajectory_logs.jsonl (257건)
💾 결과 저장: results/phi/mbpp/code_then_plan/summary.json
💾 결과 저장: results/phi/mbpp/code_then_plan/failure_examples.json
📝 failure_examples: 13개 유형 저장됨

plan_repair
============================================================
📊 결과 요약
  총 문제: 257
  성공: 207
  실행 성공: 225
  success@20: 0.8054
  execution_success_rate: 0.8755
  conditional_success: 0.9200
  AUSC: 0.7628
============================================================
  code_failed: 0
  define_test_failed: 0
  run_test_failed: 0
============================================================
  plan 사용: 92/257 (35.8%)
  repair 사용: 73/257 (28.4%)
  planning 복구 성공: 42/92 (45.7%)
  repair 복구 성공: 9/73 (12.3%)
============================================================
  [planning-cycle] 사용: 384 cycles (768 calls), 성공: 33/384 (8.6%)
  [repair-call] 사용: 351 calls, 성공: 9/351 (2.6%)
  [call-level] plan→repair 성공: 9/351
💾 결과 저장: results/phi35mini/mbpp/code_then_plan_repair/step_logs.jsonl (1376건)
💾 결과 저장: results/phi35mini/mbpp/code_then_plan_repair/trajectory_logs.jsonl (257건)
💾 결과 저장: results/phi35mini/mbpp/code_then_plan_repair/summary.json
💾 결과 저장: results/phi35mini/mbpp/code_then_plan_repair/failure_examples.json
📝 failure_examples: 13개 유형 저장됨

proposed_v1
============================================================
📊 결과 요약
  총 문제: 257
  성공: 205
  실행 성공: 224
  success@20: 0.7977
  execution_success_rate: 0.8716
  conditional_success: 0.9152
  AUSC: 0.7722
============================================================
  code_failed: 33
  define_test_failed: 0
  run_test_failed: 19
============================================================
  [problem-level] plan 사용: 75/257 (29.2%)
  [problem-level] plan 복구 성공: 17/75 (22.7%)
  [problem-level] repair 사용: 80/257 (31.1%)
  [problem-level] repair 복구 성공: 15/80 (18.8%)
  [call-level] planning-cycle 사용: 292 cycles (584 calls), 성공: 17/292 (5.8%)
  [call-level] repair 호출: 382, 성공: 15/382 (3.9%)
============================================================
💾 결과 저장: results/phi35mini/mbpp/proposed_v1/summary.json
💾 결과 저장: results/phi35mini/mbpp/proposed_v1/step_logs.jsonl (1223건)
💾 결과 저장: results/phi35mini/mbpp/proposed_v1/trajectory_logs.jsonl (257건)
✅ policy_loop 완료
DEBUG:httpcore.connection:close.started
DEBUG:httpcore.connection:close.complete

============================================================
📊 결과 요약
  총 문제: 257
  성공: 206
  실행 성공: 217
  success@20: 0.8016
  execution_success_rate: 0.8444
  conditional_success: 0.9493
  AUSC: 0.7796
============================================================
  code_failed: 40
  define_test_failed: 0
  run_test_failed: 11
============================================================
  input_tokens min/avg/max: 118 / 451.0 / 1138
  output_tokens min/avg/max: 31 / 185.2 / 512
============================================================
  [problem-level] plan 사용: 83/257 (32.3%)
  [problem-level] plan 복구 성공: 25/83 (30.1%)
  [problem-level] repair 사용: 71/257 (27.6%)
  [problem-level] repair 복구 성공: 16/71 (22.5%)
  [call-level] plan 호출 누적: 405, 성공: 25/405 (6.2%)
  [call-level] repair 호출 누적: 222, 성공: 16/222 (7.2%)
============================================================
💾 결과 저장: results/phi35mini/mbpp/proposed_v2/summary.json
💾 결과 저장: results/phi35mini/mbpp/proposed_v2/step_logs.jsonl (1289건)
💾 결과 저장: results/phi35mini/mbpp/proposed_v2/trajectory_logs.jsonl (257건)
✅ policy_loop 완료