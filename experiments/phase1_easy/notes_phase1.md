# 📘 Phase 1: Easy Code Generation (HumanEval)
## 1. 실험 목적
Phase 1의 목표는 다음과 같다:
- sLM 기반 코드 생성에서 orchestration의 효과를 검증
- 다양한 orchestration primitive가 성능을 실제로 향상시키는지, 어떤 방식으로 향상시키는지를 분석

> Q. 단순 single-shot generation 대비, orchestration은 어떤 방식으로 성능을 개선하는가?   
> 그리고 서로 다른 orchestration 방식은 어떤 종류의 문제를 해결하는가?

## 2. 실험 설정
### Set-up :
- (Datasets) HumanEval
    - Hugging Face(2021) : openai/openai_humaneval
    - Task : 함수 단위 코드 생성
    - **164 problems**
    - test-based evaluation(pass/fail)
- (BaseModel) Qwen/Qwen2.5-Coder-7B-Instruct
    - temperature = 0.0
    - max_new_tokens = 512

### 비교 방법 : 
본 실험에서는 동일한 base model (Qwen2.5-Coder-7B-Instruct)과 동일한 task(HumanEval)에 대해 다음과 같은 orchestration 설정을 비교한다.

1. **Single-shot (baseline)**
   - 단일 모델 호출로 코드 생성

2. **Repair loop**
   - 초기 생성 실패 시, execution feedback을 기반으로 반복 수정

3. **Planner-Coder**
   - (Planner) 문제 해결 계획 생성  
   - (Coder) 계획을 기반으로 코드 생성
   > 같은 모델, 다른 프롬프트

### 평가 지표
본 연구에서는 기존 코드 생성 평가 지표와, orchestration 효과를 분석하기 위한 확장 지표를 함께 사용한다.

#### (1) 기존 지표 (Standard Metrics)
- **Pass@1**: test case를 통과한 문제 비율

#### (2) 보조 분석 지표 (Diagnostic Metrics)
- **Latency (s)**: 문제당 평균 실행 시간
- **Error Type Distribution**: 실패 유형 분포

#### (3) 확장 지표 (Orchestration-aware Metrics)
- **Gain (vs Single)**: baseline(single)에서 실패했지만 해당 방법에서는 성공한 문제 수
- **Repair-only Gain**: repair만 해결 가능한 문제 수
- **Planner-only Gain**: planner-coder만 해결 가능한 문제 수

## 3. 결과

성능 순서 : Repair > Planner-Coder > Single

| Method        | Pass@1 | Pass | *Fail | Avg Latency (s) | Gain vs Single | Repair-only Gain | Planner-only Gain |
|---------------|--------|------|------|------------------|----------------|------------------|-------------------|
| Single        | 0.7195 | 118  | 46   | 2.117            | -              | -                | -                 |
| Repair        | 0.8415 | 138  | 26   | 3.390            | +20            | 2                | -                 |
| Planner-Coder | 0.8171 | 134  | 30   | 1.799            | +29            | -                | 11                |

- Pass : test case를 모두 통과한 경우(실행성공 + 정답)
- Fail : test case를 통과하지 못한 모든 경우(실행에러, 오답, 문법에러, Total - Pass - Timeout 등)

### Error Type Distribution

| Method        | *Fail (Total) | Assertion | Syntax | Name | Others |
|---------------|--------------|----------|--------|------|--------|
| Single        | 46           | 28       | 13     | 3    | 2      |
| Repair        | 26           | 11       | 11     | 4    | 0      |
| Planner-Coder | 30           | 18       | 6      | 4    | 2      |

<details>
<summary><strong>Error Type 정의</strong></summary>

본 실험에서는 실행 결과를 바탕으로 실패 유형을 다음과 같이 분류하였다.

| Error Type        | 정의 | 실패 유형 해석 |
|------------------|------|----------------|
| Assertion Error  | assert 조건을 만족하지 못한 경우 (정답 불일치) | Semantic failure (문제 해결 실패) |
| Syntax Error     | Python 문법 오류로 코드 실행 불가 | Formatting failure (출력 구조 붕괴) |
| Name Error       | 정의되지 않은 변수/함수를 참조 | Grounding failure (변수/함수 정합성 문제) |
| Type Error       | 잘못된 타입 연산 또는 함수 사용 | Type handling failure |
| Value Error      | 유효하지 않은 값 처리 실패 | Input/value handling failure |
| Runtime Error    | 실행 중 발생하는 기타 오류 | Execution failure (실행 안정성 문제) |
| Others           | 위에 속하지 않는 기타 오류 | 기타 |

</details>