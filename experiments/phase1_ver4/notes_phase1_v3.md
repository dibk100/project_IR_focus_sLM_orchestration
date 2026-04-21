# 📘 Phase 1: Easy Code Generation (HumanEval, MBPP)
## Architecture 
```
Problem
  -> Prompt Builder
  -> Orchestration Policy
      -> Generator
      -> (optional) Decomposer
      -> (optional) Verifier
      -> (optional) Repair / Retry
      -> (optional) Selector
  -> Final Candidate
  -> Executor / Test Runner
  -> Metrics Logger
```

## 1. 실험 목적
- 쉬운 코드 생성 환경에서 sLM orchestration의 기본 성능 증폭 효과를 확인한다.
- 비구조적 다중 시도(retry, best-of-N)와 구조적 orchestration(repair, planning)의 차이를 분리해 분석한다.
- 동일하거나 유사한 추론 budget 하에서 어떤 구조가 가장 비용 효율적으로 성능을 개선하는지 평가한다.

## 2. 실험 설정
### Datasets
- HumanEval (OpenAI, 2021) : **164 problems**
   - HumanEval = main mechanism
- MBPP (Google Research, 2021) : **374 problems**
   - MBPP = robustness check

### Datasets(add)
- BigCodeBench (ICLR 2025) - BigCodeBench-Instruct : **1,140 problems**
- LiveCodeBench (ICLR 2024) - v6 : **1,055 problems**
- ClassEval (ICSE, 2024) : **100 problems**

### Base Model
- Qwen/Qwen2.5-Coder-7B-Instruct
    - baseline = Single-Shot
    - temperature = 0.2~0.4 (0.2로 먼저 실험)
    - max_new_tokens = 512
    - N : 3
    - call : 3(1,2loop)

### primitive
- Generator (G): 코드 초안 생성
- Retry (R): 실패 시 새로 생성
- Verifier (V): 실행/테스트/정적 체크로 검증
   - V_exec: 실행 가능 여부 확인
      - 실패 시 EXEC_FAIL:<error_type>
   - V_test: 테스트 통과 여부 확인
      - 실패 시 TEST_FAIL:<error_type>
      - 테스트 통과 : PASS

- Decomposer (D): 문제를 하위 단계로 분해
- Selector (S): 여러 후보 중 선택
   - Best-of-N selector : 
      - 실행 가능한 후보 우선 선택 -> 실행 가능한 후보들에 대해 TEST 실행 -> 모두 TEST 실행했다면 길이가 짧은 결과를 출력/ 실패했다면 fail
      - 모두 실행 불가능 -> fail

### Methods
- Single-shot(G) : 1회 생성
- Retry-only(G -> R) : 프롬프트에 입력, 이전코드
- Best-of-N(G x N -> S) : N개 후보를 독립 생성한 뒤 후처리로 선택하는 pass-seeking policy
- Repair loop(G -> V -> R)
   - 프롬프트에 (입력, 이전코드, 에러메세지(exec fail, test fail))가 넣어짐.
   - 추후에 결과 분석을 위해 각 loop마다 어떤 단계(exec fail, test faill, call faill 등)이었던게 해결되었는지 확인이 필요
- Planner-Coder(D → G)
   - **Constrained Planner-Coder(step 수 최대 5 제한, 각 step 최대 20 tokens, 최종 plan 총 길이 최대 100~150 tokens)**
   - Unconstrained Planner-Coder(자율 서술형 plan : 나중에 ablation으로 하기)
- Planner + Repair(D -> G -> V -> R) 

```
Single-shot: Problem -> G -> Evaluate
Retry-only: Problem -> G -> if fail, R -> Evaluate
Best-of-N: Problem -> G x N -> S -> Evaluate
Repair loop: Problem -> G -> V -> R -> Evaluate
Planner-Coder: Problem -> D -> G -> Evaluate
Planner+Repair: Problem -> D -> G -> V -> R -> Evaluate
```

#### 비교 대상:
- Level 1 (기본)
    - Single-shot (fixed compute)
- Level 2 (비구조적)
    - Retry
    - Best-of-N
- Level 3 (구조적)
    - Repair
    - Planner

### Metrics
#### Performance
- Pass@1 (= overall pass rate) : 정답 비율
- Execution success rate (= structural success) : 코드가 실행 가능한 비율
- Conditional pass rate : P(pass | executable) : 실행 가능한 것 중 정답 비율

#### Cost
- Total token usage : 입력/출력/누적 토큰
- Latency
- Calls per problem : 문제 하나 해결에 평균 몇 단계가 필요한지

#### Efficiency
- Gain vs single
- Cost-normalized gain :
   - ΔPass@1 / 1k tokens
   - ΔPass@1 / extra second
   - ΔPass@1 / extra call

#### Scaling Behavior under Budget
- Solve rate at 1 / 2 / 4 / 8 call budgets
- Solve rate under matched token budgets
- Solve rate under matched latency budgets

### Analysis
- Raw comparison
- Call-matched comparison
- Token-matched comparison
- Failure type breakdown
- HumanEval main / MBPP robustness 분리 보고
   
## 3.실험 설계
### Group A: Minimal baselines

- Single-shot
- Retry-only
- Best-of-N(same budget 기준으로 retry-only와 대응)

### Group B: Structured orchestration
- Repair loop
- Planner-Coder
- Planner + Repair

## 4. 실험 로그 스키마(JSON 로그 설계)
### configs_yaml 설정
```
# Phase 1: HumanEval Single-Shot Baseline
# 사용법:
# PYTHONPATH=. python -m src.orchestration.single experiments/phase1_easy/configs/humaneval_single.yaml

run:
  run_id: "phase1_qwen25coder7b_humaneval_single"
  seed: 42

dataset:
  name: "humaneval"
  num_samples: 164

model:
  name: "Qwen/Qwen2.5-Coder-7B-Instruct"
  max_new_tokens: 512
  temperature: 0.2

method:
  name: "single_shot"

budget:
  max_calls: 1                #  실제 모델 호출 횟수

output:
  dir: "results/phase1_easy/humaneval/single"

logging:
  save_step_level: true
  save_trajectory_level: true
  save_problem_summary: true
  save_run_analysis: true
  save_code: true
```

### 4-1. Step-level (각 call 단위)
모델이 한 번 호출될 때마다 생성되는 로그 (가장 granular한 단위)

```
{
  "run_id": "phase1_qwen25coder7b_humaneval_single_0300234",
  "dataset": "humanEval",
  "problem_id": "HumanEval/23",
  "method": "best_of_n",                  # "single_shot", "retry_only", "best_of_n", "repair_loop", "planner_coder"
  "trajectory_id": "prob23_run0",         # 하나의 문제 실행 흐름 ID : "problem_id + run index"
  "step_id": 0,                           # 해당 trajectory 내 step 순서
  "call_index": 0,                        # 실제 모델 호출 순서
  "candidate_id": 0,                      # Best-of-N에서 각 후보 구분용

  "stage": "generate",                    # "generate", "retry", "repair", "plan"
  "is_retry": false,
  "is_repair": false,
  "is_planner": false,

  "input_tokens": 412,                    # 모델에 넣은 프롬프트 전체 토큰 수
  "output_tokens": 198,                   # 모델이 생성한 전체 출력 텍스트 토큰 수
  "total_tokens": 610,
  "latency_sec": 1.84,

  "code": "...generated code string...",  # 모델이 생성한 코드 (raw string)

  "exec_ok": true,
  "test_pass": false,
  "status": "TEST_FAIL:AssertionError",
  "error_type": "AssertionError",
  "error_stage": "test",

  "tests_passed": 3,                      # 통과한 테스트 수
  "tests_total": 7,                       # 전체 테스트 수
  "code_length": 523,
   
   # Best-of-N 전용
  "selected": false,                      # 해당 후보가 최종 선택됐는지
  "selection_rank": null                  # 최종 선택된 후보 = 1
}
```

#### State 관련
- exec_ok: 실행 성공 여부
- test_pass: 테스트 통과 여부
- status:
   - "PASS"
   - "EXEC_FAIL:TypeError"
   - "TEST_FAIL:AssertionError"

#### 구조 관련
- stage:
   - "generate"
   - "retry"
   - "repair"
   - "plan"
- is_retry, is_repair, is_planner

#### 비용 분석
- input_tokens
- output_tokens
- latency_sec

#### 성능 분석
- tests_passed
- tests_total

### 4-2. Trajectory-level (문제 단위 실행 흐름)
하나의 문제에 대해 orchestration이 끝날 때까지의 전체 흐름 요약

```
{
  "run_id": "exp001",
  "dataset": "HumanEval",
  "problem_id": "HumanEval/23",
  "method": "repair_loop",
  "trajectory_id": "prob23_run0",

  "num_steps": 3,             # trajectory 내 전체 orchestration 단계 수
  "call_count": 3,            # 실제 모델 호출 횟수

  "final_status": "PASS",     # 최종 상태
  "final_tests_passed": 7,    # 최종 통과 테스트 수
  "final_tests_total": 7,     # 전체 테스트 수

  "total_tokens": 1820,       # 모든 step의 토큰 합
  "total_latency": 5.12,      # 총 실행 시간

  "num_exec_fail": 1,         # exec 실패 횟수
  "num_test_fail": 1,         # test 실패 횟수

  "transition_path": [        # 상태 변화 흐름
   "EXEC_FAIL:TypeError",
   "TEST_FAIL:AssertionError",
   "PASS"
   ],
}
```

#### transition_path
가장 중요한 결과 정보

#### budget 추적 가능
예시
``` 
  "budget_used": {
    "tokens": 1820,
    "calls": 3,
    "latency": 5.12
  }
```

### 4-3. Problem-level summary
하나의 실험(run_id)에 대한 전체 성능 요약 : ipynb로 코드 구현하기
```
{
  "run_id": "exp001",
  "dataset": "HumanEval",
  "method": "repair_loop",

  "total_problems": 164,
  "num_pass": 87,

  "pass_at_1": 0.53,
  "execution_success_rate": 0.78,
  "conditional_pass": 0.68,

  "avg_tokens": 1320,
  "avg_latency": 3.1,
  "avg_calls": 2.4
}
```

### 4-4. Run-level analysis summary
분석 결과 파일로 나중에 ipynb로 코드 구현하기
```
{
  "run_id": "exp001",
  "dataset": "HumanEval",
  "method": "repair_loop",

  "transition_counts": {
    "EXEC_FAIL->PASS": 12,
    "EXEC_FAIL->TEST_FAIL": 8,
    "TEST_FAIL->PASS": 20,
    "TEST_FAIL->TEST_FAIL": 15
  },

  "failure_type_counts": {
    "EXEC_FAIL:SyntaxError": 11,
    "EXEC_FAIL:TypeError": 7,
    "TEST_FAIL:AssertionError": 29
  }
}
```