# Qwen2.5-Coder-7B-Instruct / HumanEval 결과 정리

model : Qwen
dataset : humaneval

Single
============================================================
📊 결과 요약
  총 문제: 164
  성공: 116
  실행 성공: 145
  success@1: 0.7073
  execution_success_rate: 0.8841
  conditional_success: 0.8000
============================================================
  code_failed: 19
  define_test_failed: 0
  run_test_failed: 29
============================================================
💾 결과 저장: results/qwen25coder7b/humaneval/single/step_logs.jsonl (164건)
💾 결과 저장: results/qwen25coder7b/humaneval/single/trajectory_logs.jsonl (164건)
💾 결과 저장: results/qwen25coder7b/humaneval/single/summary.json
💾 결과 저장: results/qwen25coder7b/humaneval/single/analysis.json

Repair
============================================================
📊 결과 요약
  총 문제: 164
  성공: 138
  실행 성공: 144
  success@20: 0.8415
  execution_success_rate: 0.8780
  conditional_success: 0.9583
============================================================
  code_failed: 20
  define_test_failed: 0
  run_test_failed: 6
============================================================
💾 결과 저장: results/qwen25coder7b/humaneval/repair/step_logs.jsonl (718건)
💾 결과 저장: results/qwen25coder7b/humaneval/repair/trajectory_logs.jsonl (164건)
💾 결과 저장: results/qwen25coder7b/humaneval/repair/summary.json
💾 결과 저장: results/qwen25coder7b/humaneval/repair/failure_examples.json
📝 failure_examples: 10개 유형 저장됨

Planning
============================================================
📊 결과 요약
  총 문제: 164
  성공: 152
  실행 성공: 163
  success@20: 0.9268
  execution_success_rate: 0.9939
  conditional_success: 0.9325
  AUSC: 0.8927
============================================================
  code_failed: 0
  define_test_failed: 0
  run_test_failed: 0
============================================================
  plan 사용: 48/164 (29.3%)
  plan 복구 성공: 36/48 (75.0%)
============================================================
💾 결과 저장: results/qwen25coder7b/humaneval/code_then_plan/step_logs.jsonl (492건)
💾 결과 저장: results/qwen25coder7b/humaneval/code_then_plan/trajectory_logs.jsonl (164건)
💾 결과 저장: results/qwen25coder7b/humaneval/code_then_plan/summary.json
💾 결과 저장: results/qwen25coder7b/humaneval/code_then_plan/failure_examples.json
📝 failure_examples: 9개 유형 저장됨

Planning+Repair
============================================================
📊 결과 요약
  총 문제: 164
  성공: 151
  실행 성공: 160
  success@20: 0.9207
  execution_success_rate: 0.9756
  conditional_success: 0.9437
  AUSC: 0.8936
============================================================
  code_failed: 0
  define_test_failed: 0
  run_test_failed: 0
============================================================
  plan 사용: 49/164 (29.9%)
  repair 사용: 20/164 (12.2%)
  planning 복구 성공: 36/49 (73.5%)
  repair 복구 성공: 2/20 (10.0%)
============================================================
  [planning-cycle] 사용: 119 cycles (238 calls), 성공: 34/119 (28.6%)
  [repair-call] 사용: 85 calls, 성공: 2/85 (2.4%)
  [call-level] plan→repair 성공: 2/85
💾 결과 저장: results/qwen25coder7b/humaneval/code_then_plan_repair/step_logs.jsonl (487건)
💾 결과 저장: results/qwen25coder7b/humaneval/code_then_plan_repair/trajectory_logs.jsonl (164건)
💾 결과 저장: results/qwen25coder7b/humaneval/code_then_plan_repair/summary.json
💾 결과 저장: results/qwen25coder7b/humaneval/code_then_plan_repair/failure_examples.json
📝 failure_examples: 11개 유형 저장됨

proposed_v1
📊 결과 요약
  총 문제: 164
  성공: 154
  실행 성공: 163
  success@20: 0.9390
  execution_success_rate: 0.9939
  conditional_success: 0.9448
  AUSC: 0.9064
============================================================
  code_failed: 1
  define_test_failed: 0
  run_test_failed: 9
============================================================
  [problem-level] plan 사용: 33/164 (20.1%)
  [problem-level] plan 복구 성공: 21/33 (63.6%)
  [problem-level] repair 사용: 22/164 (13.4%)
  [problem-level] repair 복구 성공: 12/22 (54.5%)
  [call-level] planning-cycle 사용: 76 cycles (152 calls), 성공: 21/76 (27.6%)
  [call-level] repair 호출: 77, 성공: 12/77 (15.6%)
============================================================
💾 결과 저장: results/qwen25coder7b/humaneval/proposed_v1/summary.json
💾 결과 저장: results/qwen25coder7b/humaneval/proposed_v1/step_logs.jsonl (393건)
💾 결과 저장: results/qwen25coder7b/humaneval/proposed_v1/trajectory_logs.jsonl (164건)
✅ policy_loop 완료


============================================================
📊 결과 요약
  총 문제: 164
  성공: 152
  실행 성공: 162
  success@20: 0.9268
  execution_success_rate: 0.9878
  conditional_success: 0.9383
  AUSC: 0.8006
============================================================
  code_failed: 2
  define_test_failed: 0
  run_test_failed: 10
============================================================
  [problem-level] plan 사용: 88/164 (53.7%)
  [problem-level] plan 복구 성공: 76/88 (86.4%)
  [problem-level] repair 사용: 133/164 (81.1%)
  [problem-level] repair 복구 성공: 45/133 (33.8%)
  [call-level] planning-cycle 사용: 187 cycles (374 calls), 성공: 76/187 (40.6%)
  [call-level] repair 호출: 247, 성공: 45/247 (18.2%)
============================================================
💾 결과 저장: results/qwen25coder7b/humaneval/proposed_v3/summary.json
💾 결과 저장: results/qwen25coder7b/humaneval/proposed_v3/step_logs.jsonl (785건)
💾 결과 저장: results/qwen25coder7b/humaneval/proposed_v3/trajectory_logs.jsonl (164건)
✅ policy_loop 완료


============================================================
📊 결과 요약
  총 문제: 164
  성공: 149
  실행 성공: 156
  success@20: 0.9085
  execution_success_rate: 0.9512
  conditional_success: 0.9551
  AUSC: 0.8003
============================================================
  code_failed: 8
  define_test_failed: 0
  run_test_failed: 7
============================================================
  [problem-level] plan 사용: 89/164 (54.3%)
  [problem-level] plan 복구 성공: 74/89 (83.1%)
  [problem-level] repair 사용: 133/164 (81.1%)
  [problem-level] repair 복구 성공: 44/133 (33.1%)
  [call-level] planning-cycle 사용: 130 cycles (260 calls), 성공: 74/130 (56.9%)
  [call-level] repair 호출: 239, 성공: 44/239 (18.4%)
============================================================
💾 결과 저장: results/qwen25coder7b/humaneval/proposed_v3_2/summary.json
💾 결과 저장: results/qwen25coder7b/humaneval/proposed_v3_2/step_logs.jsonl (663건)
💾 결과 저장: results/qwen25coder7b/humaneval/proposed_v3_2/trajectory_logs.jsonl (164건)
✅ policy_loop 완료