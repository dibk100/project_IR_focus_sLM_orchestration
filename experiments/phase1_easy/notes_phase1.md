# 📘 Phase 1: Easy Code Generation (HumanEval)
## 1. 실험 목적
- sLM 기반 코드 생성에서 orchestration이 실제로 성능 향상을 유도하는지 검증
- orchestration을 구성하는 primitive (R, V, D)의 역할을 분해하고, 각 primitive 및 그 상호작용이 성능 향상에 어떻게 기여하는지 분석

> Q. 단순 single-shot generation 대비, orchestration은 어떤 방식으로 성능을 개선하는가?
> Q. 각 primitive (R, V, D)는 성능 향상에 어떤 방식으로 기여하며, 어떤 한계를 가지는가?
> Q. 특히, primitive 간 결합(R + V)은 성능에 어떤 영향을 미치는가?
>> Q. imperfect한 verification signal은 refinement와 결합될 때 어떤 방식으로 성능 향상을 유도하는가?

## 2. 실험 설정
### Datasets
- HumanEval (OpenAI, 2021) : **164 problems**
- Task : 함수 단위 코드 생성
   - test-based evaluation(pass/fail)
### Base Model
- Qwen/Qwen2.5-Coder-7B-Instruct
    - baseline = Single-Shot
    - temperature = 0.0
    - max_new_tokens = 512

## 3. 비교 방법(Orchestraion 구성) : 

본 연구에서 정의한 primitive(G, V, R, D)는 기존 LLM 기반 코드 생성 및 agent 연구에서 공통적으로 관찰되는 문제 해결 구조를 기반으로 기능 단위로 분해한 것이다. 예를들어, iterative self‑refinement 연구에서는 생성된 output을 평가(Verification)하고 이를 바탕으로 수정(Refinement)하는 흐름이 주요 구성으로 사용되었고, compositional program synthesis 연구에서는 문제를 계획/분해(Decomposition)한 뒤 그 구조를 바탕으로 코드 생성(Generation)을 수행하는 방식이 제안되었다. 이러한 반복적 패턴을 기능 단위로 추상화한 것이 본 연구의 primitive 정의이다.

본 실험에서는 다양한 orchestration 전략을 **primitive 단위로 분해**하고, 
각 primitive 및 그 조합이 코드 생성 성능에 미치는 **기여**를 분석한다.

#### (1) Baseline
- Single-shot (G)
   - 단일 모델 호출을 통해 코드 생성 후 바로 평가
   - orchestration을 적용하지 않은 기본 설정

#### (2) Primitive-level Configurations
각 primitive의 독립적인 효과를 분석하기 위해 다음과 같은 구성을 사용한다.

- Refinement-only (G + R)
    - 이전에 생성된 코드를 기반으로 수정 수행 (feedback 없음)
    - execution feedback 없이 코드 자체만을 개선
    - self-refinement 능력 측정
    
- Verification-only (G + V)
   - 생성된 코드의 정답 여부(PASS/FAIL)만 판단
   - 코드 수정 없이 판별만 수행
   - error detection 능력 측정

#### (3) Composite Orchestration (Method-level)
primitive들을 조합한 실제 orchestration 구조를 비교한다.

- Repair loop (G + V + R)
   - 코드 생성 → 실행 → error feedback → 수정의 반복 구조
   - verification과 refinement의 결합

- Planner-Coder (D → G)
   - 문제를 자연어 계획(planning)으로 분해한 후 코드 생성 수행
   - decomposition 기반 generation 구조

## 4. 평가 지표
본 연구에서는 기존 코드 생성 평가 지표와, orchestration 효과를 분석하기 위한 확장 지표(self-make)를 함께 사용한다.

#### (1) 기존 지표 (Standard Metrics)
- **Pass@1**: test case를 통과한 문제 비율

#### (2) 보조 분석 지표 (Diagnostic Metrics)
- **Latency (s)**: 문제당 평균 실행 시간
- **Error Type Distribution**: 실패 유형 분포

#### (3) 확장 지표 (Orchestration-aware Metrics)
- **Gain (vs Single)**: baseline(single)에서 실패했지만 해당 방법에서는 성공한 문제 수
- **Repair-only Gain**: repair만 해결 가능한 문제 수
- **Planner-only Gain**: planner-coder만 해결 가능한 문제 수

## 5. 결과
### 5.1 전체 성능 비교
성능 순서 : Repair > Planner-Coder > Refinement > Single

| Method        | Pass@1 | Pass | *Fail | Avg Latency (s) | Gain vs Single | Repair-only Gain | Planner-only Gain |
|---------------|--------|------|------|------------------|----------------|------------------|-------------------|
| Single        | 0.7195 | 118  | 46   | 2.117            | -              | -                | -                 |
| Refinement    | 0.7744 | 127  | 37   | 2.859            | +9             | -                | -                 |
| Repair        | 0.8415 | 138  | 26   | 3.390            | +20            | 2                | -                 |
| Planner-Coder | 0.8171 | 134  | 30   | 1.799            | +29            | -                | 11                |

- Pass : test case를 모두 통과한 경우(실행성공 + 정답)
- Fail : test case를 통과하지 못한 모든 경우(실행에러, 오답, 문법에러, Total - Pass - Timeout 등)

### 5.2 Error Type Distribution

| Method        | Fail (Total) | Assertion | Syntax | Name | Others |
| ------------- | ------------ | --------- | ------ | ---- | ------ |
| Single        | 46           | 28        | 13     | 3    | 2      |
| Refinement    | 37           | 4         | 12     | 21   | 0      |
| Repair        | 26           | 11        | 11     | 4    | 0      |
| Planner-Coder | 30           | 18        | 6      | 4    | 2      |

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

### 5.3 Verification-only 결과
- Verifier Accuracy: 0.8049

Confusion Matrix:
|           | Pred PASS | Pred FAIL |
| --------- | --------- | --------- |
| True PASS | 114       | 4         |
| True FAIL | 28        | 18        |

Derived Metrics:
- Recall (PASS): 0.966
- Specificity (FAIL): 0.391

> 맞는 코드는 거의 다 맞다고 판단    
> 틀린 코드는 대부분 PASS로 오판

