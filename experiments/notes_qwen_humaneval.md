# Qwen2.5-Coder-7B-Instruct / HumanEval 결과 정리

| Method            |       Success | Success Rate | Execution Success | Execution Rate | Conditional Success |       AUSC | Code Failed | Run Test Failed |       Plan Usage |      Repair Usage | Recovery Success                                 |
| ----------------- | ------------: | -----------: | ----------------: | -------------: | ------------------: | ---------: | ----------: | --------------: | ---------------: | ----------------: | ------------------------------------------------ |
| Single            |     116 / 164 |       70.73% |         145 / 164 |         88.41% |              80.00% |          - |          19 |              29 |                - |                 - | -                                                |
| Repair            |     138 / 164 |       84.15% |         144 / 164 |         87.80% |              95.83% |          - |          20 |               6 |                - |         Iterative | Repair 중심                                        |
| Planning          |     152 / 164 |       92.68% |         163 / 164 |         99.39% |              93.25% |     0.8927 |           0 |               0 | 48 / 164 (29.3%) |                 - | Plan: 36 / 48 (75.0%)                            |
| Planning + Repair |     151 / 164 |       92.07% |         160 / 164 |         97.56% |              94.37% |     0.8936 |           0 |               0 | 49 / 164 (29.9%) |  20 / 164 (12.2%) | Plan: 36 / 49 (73.5%) / Repair: 2 / 20 (10.0%)   |
| Proposed v1       | **154 / 164** |   **93.90%** |         163 / 164 |         99.39% |          **94.48%** | **0.9064** |           1 |               9 | 33 / 164 (20.1%) |  22 / 164 (13.4%) | Plan: 21 / 33 (63.6%) / Repair: 12 / 22 (54.5%)  |
| Proposed v2       |     153 / 164 |       93.29% |         163 / 164 |         99.39% |              93.87% |     0.8152 |           1 |              10 | 96 / 164 (58.5%) | 109 / 164 (66.5%) | Plan: 84 / 96 (87.5%) / Repair: 40 / 109 (36.7%) |


### 추가 운영 통계

| Method            | Step Logs |         Planning Calls | Repair Calls | Token Stats                                   | 특징                       |
| ----------------- | --------: | ---------------------: | -----------: | --------------------------------------------- | ------------------------ |
| Single            |       164 |                      - |            - | -                                             | 단일 호출 baseline           |
| Repair            |       718 |                      - | 다수 iterative | -                                             | 반복 수정 기반                 |
| Planning          |       492 |   48 cycles (96 calls) |            - | -                                             | 실패 시 plan-first 전략       |
| Planning + Repair |       487 | 119 cycles (238 calls) |     85 calls | -                                             | planning 이후 repair 연계    |
| Proposed v1       |   **393** |  76 cycles (152 calls) |     77 calls | -                                             | selective orchestration  |
| Proposed v2       |       743 |              191 calls |    197 calls | in: 37 / 416.7 / 1592<br>out: 1 / 157.0 / 512 | aggressive orchestration |


### 핵심 비교 포인트
1. 최고 정확도
Proposed v1
success@20 = 93.90%
AUSC = 0.9064
가장 높은 overall efficiency-performance tradeoff
2. 최고 복구 능력
Proposed v2
planning recovery = 87.5%
repair recovery = 36.7%
orchestration 자체의 recovery capability는 strongest
3. 효율성 측면
Proposed v1
가장 적은 step logs (393)
낮은 orchestration usage
selective intervention 구조
Proposed v2
step logs 743
repair 사용률 66.5%
planning 사용률 58.5%
높은 compute consumption 대비 성능 이득은 제한적
4. 흥미로운 관찰
Proposed v2는 recovery success는 크게 증가했지만:
최종 success는 v1보다 낮음
AUSC는 크게 감소 (0.9064 → 0.8152)

이는 더 많은 orchestration이 항상 더 좋은 budget-efficiency로 이어지지 않음을 시사함.


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