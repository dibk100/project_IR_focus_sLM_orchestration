model : Qwen 
dataset : mbpp

single
============================================================
📊 결과 요약
  총 문제: 257
  성공: 135
  실행 성공: 172
  success@1: 0.5253
  execution_success_rate: 0.6693
  conditional_success: 0.7849
============================================================
  code_failed: 85
  define_test_failed: 0
  run_test_failed: 37
============================================================
💾 결과 저장: results/qwen25coder7b/mbpp/single/step_logs.jsonl (257건)
💾 결과 저장: results/qwen25coder7b/mbpp/single/trajectory_logs.jsonl (257건)
💾 결과 저장: results/qwen25coder7b/mbpp/single/summary.json
💾 결과 저장: results/qwen25coder7b/mbpp/single/analysis.json

repair
============================================================
📊 결과 요약
  총 문제: 257
  성공: 213
  실행 성공: 256
  success@20: 0.8288
  execution_success_rate: 0.9961
  conditional_success: 0.8320
  AUSC: 0.8088
============================================================
  code_failed: 1
  define_test_failed: 0
  run_test_failed: 43
============================================================
💾 결과 저장: results/qwen25coder7b/mbpp/repair/failure_examples.json
📝 failure_examples: 8개 유형 저장됨

planning
============================================================
📊 결과 요약
  총 문제: 257
  성공: 213
  실행 성공: 256
  success@20: 0.8288
  execution_success_rate: 0.9961
  conditional_success: 0.8320
  AUSC: 0.7911
============================================================
  code_failed: 0
  define_test_failed: 0
  run_test_failed: 0
============================================================
  plan 사용: 118/257 (45.9%)
  plan 복구 성공: 74/118 (62.7%)
============================================================
💾 결과 저장: results/qwen25coder7b/mbpp/code_then_plan/step_logs.jsonl (1243건)
💾 결과 저장: results/qwen25coder7b/mbpp/code_then_plan/trajectory_logs.jsonl (257건)
💾 결과 저장: results/qwen25coder7b/mbpp/code_then_plan/summary.json
💾 결과 저장: results/qwen25coder7b/mbpp/code_then_plan/failure_examples.json
📝 failure_examples: 13개 유형 저장됨

planning+repair
============================================================
📊 결과 요약
  총 문제: 257
  성공: 214
  실행 성공: 216
  success@20: 0.8327
  execution_success_rate: 0.8405
  conditional_success: 0.9907
  AUSC: 0.7852
============================================================
  code_failed: 0
  define_test_failed: 0
  run_test_failed: 0
============================================================
  plan 사용: 124/257 (48.2%)
  repair 사용: 55/257 (21.4%)
  planning 복구 성공: 81/124 (65.3%)
  repair 복구 성공: 1/55 (1.8%)
============================================================
  [planning-cycle] 사용: 0 cycles (0 calls), 성공: 0/0
  [repair-call] 사용: 0 calls, 성공: 0/0
  [call-level] plan→repair 성공: 0/0
💾 결과 저장: results/qwen25coder7b/mbpp/code_then_plan_repair/failure_examples.json
📝 failure_examples: 11개 유형 저장됨

Proposed v1
============================================================
📊 결과 요약
  총 문제: 257
  성공: 221
  실행 성공: 250
  success@20: 0.8599
  execution_success_rate: 0.9728
  conditional_success: 0.8840
  AUSC: 0.7765
============================================================
  code_failed: 7
  define_test_failed: 0
  run_test_failed: 29
============================================================
  [problem-level] plan 사용: 0/0
  [problem-level] plan 복구 성공: 0/0
  [problem-level] repair 사용: 0/0
  [problem-level] repair 복구 성공: 0/0
  [call-level] planning-cycle 사용: 0 cycles (0 calls), 성공: 0/0
  [call-level] repair 호출: 0, 성공: 0/0
============================================================
✅ policy_loop 완료

Proposed v2
============================================================
📊 결과 요약
  총 문제: 257
  성공: 223
  실행 성공: 253
  success@20: 0.8677
  execution_success_rate: 0.9844
  conditional_success: 0.8814
  AUSC: 0.7827
============================================================
  code_failed: 4
  define_test_failed: 0
  run_test_failed: 30
============================================================
  input_tokens min/avg/max: 91 / 298.5 / 945
  output_tokens min/avg/max: 3 / 156.4 / 512
============================================================
  [problem-level] plan 사용: 119/257 (46.3%)
  [problem-level] plan 복구 성공: 85/119 (71.4%)
  [problem-level] repair 사용: 96/257 (37.4%)
  [problem-level] repair 복구 성공: 2/96 (2.1%)
  [call-level] plan 호출 누적: 394, 성공: 85/394 (21.6%)
  [call-level] repair 호출 누적: 244, 성공: 2/244 (0.8%)
============================================================
💾 결과 저장: results/qwen25coder7b/mbpp/proposed_v2/summary.json
💾 결과 저장: results/qwen25coder7b/mbpp/proposed_v2/step_logs.jsonl (1289건)
💾 결과 저장: results/qwen25coder7b/mbpp/proposed_v2/trajectory_logs.jsonl (257건)
✅ policy_loop 완료