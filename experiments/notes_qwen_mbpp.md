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

============================================================
📊 결과 요약
  총 문제: 257
  성공: 214
  실행 성공: 257
  success@20: 0.8327
  execution_success_rate: 1.0000
  conditional_success: 0.8327
  AUSC: 0.8136
============================================================
  code_failed: 0
  define_test_failed: 0
  run_test_failed: 43
============================================================
💾 결과 저장: results/qwen25coder7b/mbpp/repair/step_logs.jsonl (1172건)
💾 결과 저장: results/qwen25coder7b/mbpp/repair/trajectory_logs.jsonl (257건)
💾 결과 저장: results/qwen25coder7b/mbpp/repair/summary.json
💾 결과 저장: results/qwen25coder7b/mbpp/repair/failure_examples.json
📝 failure_examples: 10개 유형 저장됨

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
=============================================================
📊 결과 요약
  총 문제: 257
  성공: 216
  실행 성공: 218
  success@20: 0.8405
  execution_success_rate: 0.8482
  conditional_success: 0.9908
  AUSC: 0.7883
============================================================
  code_failed: 0
  define_test_failed: 0
  run_test_failed: 0
============================================================
  plan 사용: 118/257 (45.9%)
  repair 사용: 58/257 (22.6%)
  planning 복구 성공: 77/118 (65.3%)
  repair 복구 성공: 0/58 (0.0%)
============================================================
  [planning-cycle] 사용: 361 cycles (722 calls), 성공: 77/361 (21.3%)
  [repair-call] 사용: 284 calls, 성공: 0/284 (0.0%)
  [call-level] plan→repair 성공: 0/284
💾 결과 저장: results/qwen25coder7b/mbpp/code_then_plan_repair/step_logs.jsonl (1263건)
💾 결과 저장: results/qwen25coder7b/mbpp/code_then_plan_repair/trajectory_logs.jsonl (257건)
💾 결과 저장: results/qwen25coder7b/mbpp/code_then_plan_repair/summary.json
💾 결과 저장: results/qwen25coder7b/mbpp/code_then_plan_repair/failure_examples.json
📝 failure_examples: 14개 유형 저장됨

============================================================
📊 결과 요약
  총 문제: 257
  성공: 221
  실행 성공: 253
  success@20: 0.8599
  execution_success_rate: 0.9844
  conditional_success: 0.8735
  AUSC: 0.7794
============================================================
  code_failed: 4
  define_test_failed: 0
  run_test_failed: 32
============================================================
  [problem-level] plan 사용: 123/257 (47.9%)
  [problem-level] plan 복구 성공: 86/123 (69.9%)
  [problem-level] repair 사용: 99/257 (38.5%)
  [problem-level] repair 복구 성공: 3/99 (3.0%)
  [call-level] planning-cycle 사용: 290 cycles (580 calls), 성공: 86/290 (29.7%)
  [call-level] repair 호출: 290, 성공: 3/290 (1.0%)
============================================================
💾 결과 저장: results/qwen25coder7b/mbpp/proposed_v1/summary.json
💾 결과 저장: results/qwen25coder7b/mbpp/proposed_v1/step_logs.jsonl (1127건)
💾 결과 저장: results/qwen25coder7b/mbpp/proposed_v1/trajectory_logs.jsonl (257건)
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