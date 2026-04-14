# project_IR_focus_sLM_orchestration 🚀
- **Type**: 개인 연구 프로젝트 (Independent Research)
- **Area** : sLM orchestration, 구조 설계 연구
- **Main Research Questions** : 작은 모델의 한계를 agentic orchestration으로 얼마나 보완할 수 있는가? scale 부족을 system design으로 대체할 수 있는가?    
*오케스트레이션 구조란, 작은 모델을 여러 역할과 단계로 나누어 반복적으로 사용하는 실행 설계이며, 이를 통해 단일 호출의 한계를 극복하는 시스템적 방법이다

## Notes & Idea
- 평가를 decomposition
<details>
<summary><strong> 실험 metric 설계 </strong></summary>

## pass@1 : 
- 문제 당 1개 생성했을 때, 성공 확률   
pass@1 = pass = structure 성공 + semantics 성공

하지만 아래와 같은 실패 구조를 가지고 있다.
```
fail →
    ├─ exec 실패 → structural failure
    └─ assert 실패 → semantic failure
```

지표를 3가지로 쪼개보고자 함.   
## IDEA
(1) execution success rate : **strutural**   
exec(code) 성공 비율(코드 실행)

(2) pass rate : **overall**   
test 통과 비율(코드 실행 + test 성공)

(3) conditional pass : **semantic**   
pass/exec_success   
실행 가능한 코드 중에서, 실제로 정답인 비율

```
전체 시도
│
├─ execution failure
│   ├─ code stage failure
│   └─ setup stage failure
│
└─ execution success
    │
    ├─ semantic failure (assert fail)
    └─ pass
```


orchestration → exec success ↑ ? (구조 개선)   
orchestration → pass ↑ ? (전체 성능)   
orchestration → conditional pass ↑ ? (의미 개선)   

</details>




## Overview 📝
본 레포지토리는 소형 언어 모델(sLM)의 성능을 향상시키기 위한 agentic orchestration 전략을 연구한다.
대형 언어 모델이 모델 크기(scale)에 의존하는 것과 달리, 본 연구는 시스템 수준의 orchestration 설계가 작은 모델의 성능을 얼마나 끌어올릴 수 있는지를 탐구한다.    

> Final Direction : sLM-agent가 LLM과의 성능 격차를 얼마나, 어떤 구조로 줄일 수 있는가   

이전 연구는 orchestation의 동작을 failure를 활용하여 하고자 했음. (failure 목적)   
현재 연구는 failure는 수단으로 활용될 뿐, 어떤 환경에서 orchestration이 더 필요하고 효과적인지 탐구하려고 함.

## Objective
- sLM에서 어떤 orchestration 구조가 성능을 가장 효과적으로 증폭시키는지 규명
- system design으로 scale을 얼마나 대체할 수 있는지를 정량적으로 규명

## Research Questions
RQ1: sLM에서 orchestration은 single-shot 대비 성능을 얼마나 향상시키는가?
RQ2: 동일하거나 유사한 inference budget 하에서, 어떤 orchestration 구조가 가장 높은 성능 향상을 제공하는가?
RQ3: 각 orchestration 구조는 오류 유형 중 무엇을 주로 줄이는가?
- execution failure
- logical failure
- spec misunderstanding

## Methodology
- Task: 코드 생성 및 수정 (Code Generation / Editing)
    - HumanEval, MBPP: **기본 작동 여부 확인용**
    - HumanEval Pro, MBPP Pro: **난이도 상승 시 robustness 확인용**
    - (고민)SWE-bench-lite subset: **확장 가능성 점검용**
- Task 선정 이유: 코드 생성 태스크는 결과를 정량적으로 평가할 수 있음,오케스트레이션의 효과를 안정적으로 측정하기에 적합한 테스트베드로 판단함.
- 접근 방식:
    - 다양한 orchestration 구조를 설계하고, 각 구조가 sLM 성능을 얼마나 증폭시키는지와 더 큰 LLM 대비 격차를 얼마나 줄이는지를 비교 분석

## 🧪 Experimental Design
- Phase 1 : Easy Code Generation (HumanEval)
    - 목적: core 실험 설계 + baseline + orchestration 효과 확인
    - Dataset: HumanEval (164 problems), MBPP (364 problems)
    - 비교 대상:
        1. Single-shot (baseline, G)
        2. retry-only (G + R)
        3. Repair loop (G + V + R)
        4. Planner-Coder (D → G)
    - 주요 평가 지표: Pass@1, Latency, Gain vs Single, execution success rate, conditional pass
    - 연구 질문 (Phase 1 RQ):
        - 단순 single-shot generation 대비 orchestration은 어떤 방식으로 성능을 개선하는가?
        - 어떤 구조가 가장 큰 성능 증폭을 제공하는가?
        - 성능 향상 대비 비용은 어떻게 변화하는가?

- Phase 2(초안) : Harder Tasks / Structure Comparison
    - 목적: 난이도가 높은 태스크에서 Phase 1에서 확인된 orchestration 구조의 효력 검증
    - Dataset: MBPP 또는 HumanEval Pro
    - 비교 대상: Phase 1과 동일, 필요 시 hybrid orchestration 추가
    - 연구 질문 (Phase 2 RQ):
        - 난이도가 높은 태스크에서 Phase 1에서 확인된 orchestration 구조가 얼마나 일반화되는가?
        - 구조별 성능 차이는 난이도에 따라 어떻게 달라지는가?
        - Hybrid orchestration 조합은 기존 구조 대비 성능 향상에 기여하는가?

- Phase 3(초안) : Scaling / sLM vs LLM
    - 목적: Phase 1~2에서 검증된 orchestration 구조가 더 큰 모델과 비교했을 때 성능 격차를 얼마나 줄이는지 평가
    - Dataset: Phase 2 동일, 필요 시 추가 benchmark
    - 비교 대상: sLM single-shot vs sLM orchestration vs LLM single-shot
    - 연구 질문 (Phase 3 RQ):
        - sLM + orchestration은 LLM 단일 호출 대비 어느 수준까지 성능 격차를 줄일 수 있는가?
        - 비용(연산량, latency 등) 대비 성능 개선 효과는 어느 정도인가?
        - sLM baseline과 LLM 간 성능 격차 중에서 orchestration이 얼마나 이를 줄였는지를 측정

## ⭐ Research Lineage
- v1: project_IR_explore → 아이디어 탐색 및 초기 구현
- v2: project_IR_develop → 시스템 설계 및 정책 구조 개발
- v3 (현재): 본 레포지토리 → 오케스트레이션 효과에 대한 정량적 연구

## 📁 Folder Structure
ver3
```
project/
│
├── README.md
│
├── analysis/                # notebook, 결과 분석 및 정리
│   ├── data_EDA/            # step1 : 활용할 데이터 분석을 통해 TASK 구체화
│   └── .../          
│
├── datasets/          
│   ├── humaneval/
│   │   ├── raw/
│   │   └── processed/
│   ├── mbpp/
│   └── swebench_lite/
├── src/                     # 핵심 코드
│   ├── models/              # sLM, LLM wrapper
│   │   ├── base.py        
│   │   └── hf_model.py
│   ├── adapters/            # dataset별 실행 포맷 차이 처리
│   │   ├── base.py
│   │   ├── humaneval.py
│   │   └── mbpp.py
│   ├── orchestration/       # orchestration : 전략 로직 파트
│   │   ├── planner_coder.py 
│   │   ├── retry.py         
│   │   ├── repair.py        
│   │   ├── single.py        
│   │   └── verification.py
│   ├── tasks/               
│   │   ├── base.py          # task interface
│   │   ├── mbpp.py          
│   │   └── humaneval.py     # HumanEval Task Loader
│   ├── evaluation/          
│   │   ├── executor.py      # 코드 실행 및 테스트 러너
│   │   └── metrics.py       # metric 계산 : pass@1,...
│   └── utils/          
│       ├── io.py            # result 저장 유틸리티
│       └── prompting/     # 프롬프트 구성 및 코드 추출 유틸리티 : 텍스트 처리 파트
│           ├── __init__.py/
│           ├── common.py/
│           ├── humaneval.py/
│           └── mbpp.py/
│
├── experiments/
│   ├── phase1_easy/
│   │   ├── configs/
│   │   ├── run.sh
│   │   └── notes.md
│   │
│   ├── phase1_ver3/
│   │   ├── configs/
│   │   ├── run.sh
│   │   └── notes.md
│
├── results/                 
│   ├── phase1_ver3/
│   │   ├── humaneval/
│   │   │   ├── single/
│   │   │   │   ├── phase1_qwen25coder7b_humaneval_single_<숫자>/
│   │   │   │   └── ...
│   │   │   ├── retry/
│   │   │   ├── best_of_n/
│   │   │   ├── repair/
│   │   │   ├── planner_coder/
│   │   │   └── planner_repair.py/
│   │   └── mbpp/
│   │
│   └── phase1_ver2/
│       └── ...
│
├── scripts/                 # 공용 스크립트
│
├── logs/                    # raw 실행 로그
│
└── README.md
```

