# 📘 Phase 1: Orchestration Method Comparison & Analysis (HumanEval, MBPP, BigCodeBench)

## 1. 실험 목적
- 서로 다른 난이도의 코드 생성 benchmark (HumanEval, MBPP, BigCodeBench)를 대상으로 orchestration method의 성능을 비교한다.
- 동일한 실험 조건 (model, prompt, call budget)으로, 각 method의 성능(pass@1 및 추가 지표)을 평가한다.
- 그리고 Step-level 및 Trajectory-level 로그를 활용하여 각 method의 실행 과정과 failure 패턴을 분석한다.
- 반복적으로 나타나는 failure 유형과 failure stage 정보(code / define_test / run_test), 그리고 성능이 높은 method 간의 관계를 관찰하여, orchestration policy 설계를 위한 기반을 만든다.

## 2. 실험 설정
### Datasets
- HumanEval (OpenAI, 2021) : **164 problems**
   - HumanEval = main mechanism
- MBPP (Google Research, 2021) : **374 problems**
   - MBPP = robustness check
- BigCodeBench (ICLR 2025) - BigCodeBench-Instruct : **1,140 problems**

### Base Model
- Qwen/Qwen2.5-Coder-7B-Instruct
    - baseline = Single-Shot
    - temperature = 0.2~0.4 (0.2로 먼저 실험)
    - max_new_tokens = 512, 1024(bigcode)
    - call : 5

### primitive

- **Generator (G)**  
  - 주어진 문제에 대해 코드 초안을 생성하는 연산  

- **Verifier (V)**  
  - 생성된 코드의 실행 가능성과 테스트 통과 여부를 검증하는 연산  
  - 실행 단계 구분:  
    - code stage: 생성 코드 실행 실패 → EXEC_FAIL:<error_type>  
    - define_test stage: 테스트 정의 실패 → DEFINE_TEST_FAIL:<error_type>  
    - run_test stage: 테스트 실행 실패 → TEST_FAIL:<error_type>  
  - 모든 테스트 통과 시 PASS 반환  

- **Retry (R)**  
  - 이전 시도 결과를 기반으로 새로운 코드를 재생성하는 연산  

- **Repair (Rᵣ)**  
  - 실행/테스트 실패 메시지를 활용하여 기존 코드를 수정하는 연산  

- **Decomposer (D)**  
  - 문제를 하위 단계로 분해하여 해결 전략을 구성하는 연산  

(state)
- PASS
- EXEC_FAIL:<error>
- DEFINE_TEST_FAIL:<error>
- TEST_FAIL:<error>
- OTHER_FAIL:<error>

### Metrics
#### Performance
- Pass@1 (= overall pass rate)
  - 전체 문제 중 최종적으로 정답(PASS)을 맞춘 비율
  - semantic correctness를 직접 반영하는 핵심 지표

- Execution success rate (= structural success)
  - 전체 문제 중 실행 가능한 상태까지 도달한 비율
  - 기준:
    - PASS, TEST_FAIL → success
    - EXEC_FAIL, DEFINE_TEST_FAIL → failure
  - 코드가 붕괴되지 않고 실행, 테스트 정의까지 가능했는지를 측정

- Conditional pass rate : P(pass | executable)
  - 실행 가능한 코드들 중 실제로 정답인 비율
  - 계산:
    conditional_pass = pass / execution_success

#### Analysis-oriented Metrics
- **Failure Type Distribution**  
  - EX: EXEC_FAIL:SyntaxError, TEST_FAIL:AssertionError  
  - 어떤 종류의 실패가 많은지 확인  

- **Failure Stage Breakdown**  
  - code / define_test / run_test 단계별 실패 분포  

- **Transition Pattern (trajectory 기반)**  
  - EX: EXEC_FAIL → TEST_FAIL → PASS  
  - orchestration 과정에서 상태 변화 흐름 분석  

   
## 4. 실험 로그 스키마(JSON 로그 설계)
### configs_yaml 설정


### 4-1. Step-level (각 call 단위)
모델이 한 번 호출될 때마다 생성되는 로그 (가장 granular한 단위)
```
{
  "dataset": "humaneval",
  "task_id": "HumanEval/23",
  "method": "repair_loop",
  "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "attempt_idx": 0,

  "prompt": "...",
  "raw_output": "...",
  "generated_code": "...",

  "status": "TEST_FAIL:AssertionError",
  "passed": false,
  "exec_ok": true,
  "test_pass": false,
  "latency_sec": 1.84,

  "error_type": "AssertionError",
  "error_stage": "test",
  "error_message": "...",

  "tests_passed": 3,
  "tests_total": null,

  "meta": {
    "code_exec_passed": true,
    "setup_exec_passed": true,
    "test_exec_passed": false,
    "failed_stage": "run_test",
    "failed_test_index": 3,
    "raw_error_type": "AssertionError"
  }
}
```

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

  "transition_path": [        # 상태 변화 흐름
   "EXEC_FAIL:TypeError",
   "TEST_FAIL:AssertionError",
   "PASS"
   ],
}
```