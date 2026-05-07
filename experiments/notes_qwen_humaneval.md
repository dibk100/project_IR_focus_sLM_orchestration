# Qwen2.5-Coder-7B-Instruct / HumanEval 결과 정리

| Method            |   Success | Success Rate | Execution Success | Execution Rate | Conditional Success |       AUSC | Code Failed | Run Test Failed |       Plan Usage |     Repair Usage | Recovery Success                                |
| ----------------- | --------: | -----------: | ----------------: | -------------: | ------------------: | ---------: | ----------: | --------------: | ---------------: | ---------------: | ----------------------------------------------- |
| Single            | 116 / 164 |       70.73% |         145 / 164 |         88.41% |              80.00% |          - |          19 |              29 |                - |                - | -                                               |
| Repair            | 138 / 164 |       84.15% |         144 / 164 |         87.80% |              95.83% |          - |          20 |               6 |                - |        Iterative | Repair: 22 / 26 실패 복구 추정                        |
| Planning          | 152 / 164 |       92.68% |         163 / 164 |         99.39% |              93.25% |     0.8927 |           0 |               0 | 48 / 164 (29.3%) |                - | Plan: 36 / 48 (75.0%)                           |
| Planning + Repair | 151 / 164 |       92.07% |         160 / 164 |         97.56% |              94.37% |     0.8936 |           0 |               0 | 49 / 164 (29.9%) | 20 / 164 (12.2%) | Plan: 36 / 49 (73.5%) / Repair: 2 / 20 (10.0%)  |
| Proposed v1       | 154 / 164 |   **93.90%** |         163 / 164 |         99.39% |          **94.48%** | **0.9064** |           1 |               9 | 33 / 164 (20.1%) | 22 / 164 (13.4%) | Plan: 21 / 33 (63.6%) / Repair: 12 / 22 (54.5%) |

### 추가 운영 통계

| Method            | Step Logs |        Planning Cycles | Repair Calls | 특징                    |
| ----------------- | --------: | ---------------------: | -----------: | --------------------- |
| Single            |       164 |                      - |            - | 단일 호출 baseline        |
| Repair            |       718 |                      - | 다수 iterative | 반복 수정 기반              |
| Planning          |       492 |   48 cycles (96 calls) |            - | 실패 시 plan-first 전략    |
| Planning + Repair |       487 | 119 cycles (238 calls) |     85 calls | planning 이후 repair 연계 |
| Proposed v1       |       393 |  76 cycles (152 calls) |     77 calls | 선택적 orchestration     |

### 핵심 비교 포인트
최고 성능: Proposed v1
success@20 = 93.90%
AUSC = 0.9064
Planning 대비 더 적은 orchestration 사용량으로 더 높은 성능 달성
Planning only
실행 안정성(execution success) 거의 완벽 수준
repair 없이도 강력한 recovery capability 확인
Planning + Repair
planning 성능은 유지되지만 repair 효율이 매우 낮음
repair 성공률 10% 수준 → orchestration overhead 대비 효과 제한적
Proposed v1 특징
plan과 repair를 selective하게 사용
planning 사용률 감소 (29.3% → 20.1%)
repair 성공률 크게 개선 (10.0% → 54.5%)
전체 step 수도 가장 효율적(393건)