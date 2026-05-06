# project_IR_focus_sLM_orchestration 🚀
> 본 레포지토리는 개인 연구 프로젝트 **project_IR_focus_sLM_orchestration**의 Phase 1 연구를 정리한 공간.
- **Type**: 개인 연구 프로젝트 (Independent Research)
- **Area** :
  - Agentic Orchestration
  - System Design
  - Adaptive Inference Strategy
  - Reasoning Workflow Engineering
  - inference-time
- **Main Research Questions** : 
  - 작은 모델의 한계를 agentic orchestration으로 얼마나 보완할 수 있는가? scale 부족을 system design으로 대체할 수 있는가? 

## Overview 📝
본 연구 프로젝트는 작은 언어 모델(sLM)의 성능 한계를 단순한 모델 scale-up이 아닌 orchestration과 system design을 통해 극복할 수 있는지를 탐구한다.   
전체 연구는 다음과 같은 단계로 구성된다.

- Phase 1: Rule-based Policy
- Phase 2: Advanced Adaptive Policy
- Phase 3: RL-based Policy Learning


현재 레포지토리는 **Phase 1** 연구를 다룬다.

## Research Focus 🧪 
- 작은 모델의 코드 생성 한계를 orchestration으로 보완 가능한지 검증
- 동일한 inference budget 내에서 orchestration 전략 비교
- planning / repair 기반 접근의 효과 분석
- failure-aware reasoning workflow의 가능성 탐색
- adaptive strategy selection의 성능 향상 가능성 확인

## ⚙️ Experimental Setting
- **Task Type**: Code Generation, Iterative Refinement
- Common Generation Setting :
  - temperature=0.2
  - top-p=0.95
  - 최대 토큰 길이 512
  - vllm: max-model-len 4096
  - **Inference Budget**: Fixed call budget of 20
  - 생성된 코드가 성공(PASS)하면 즉시 종료(early stopping) - 문제 해결

- **Base Model**: 
  | Model                            | 특징                          |
  | -------------------------------- | --------------------------- |
  | Qwen2.5-Coder-7B-Instruct        | 코드 생성 특화 instruct 모델        |
  | microsoft/Phi-3.5-mini-instruct  | 경량 reasoning 중심 instruct 모델 |
  | meta-llama/Llama-3.1-8B-Instruct | 범용 instruct 기반 모델           |
- **Benchmarks**
  | Benchmark    | 문제 수 | 특징                     |
  | ------------ | ---- | ---------------------- |
  | HumanEval    | 164  | 함수 단위 Python 코드 생성     |
  | MBPP         | 374  | 자연어 입력, 짧은 형태의 프로그래밍 문제        |
  | BigCodeBench | 1140 | 보다 복잡하고 현실적인 코드 생성 태스크 |

- **Evaluation Metric**: 

| Metric              | Description                                 | Purpose                                      |
| ------------------- | ------------------------------------------- | -------------------------------------------- |
| Success@20          | 제한된 call budget(20) 내 최종 문제 해결 성공률          | 전체 task solving 성능 평가                        |
| Exec Success        | 생성된 코드 중 실행 가능한 코드의 비율                      | syntax validity 및 runtime stability 평가       |
| Conditional Success | 실행 가능한 코드 중 정답 코드의 비율                       | 실행 이후 실제 correctness 평가                      |
| Success@k Curve     | 각 시점 (k)까지의 누적 성공률(cumulative success rate) | recovery speed 및 orchestration efficiency 분석 |

*AUSC

## 🧩 Compared Strategies

| Method         | Description                                                            |
| -------------- | ---------------------------------------------------------------------- |
| Single-shot    | 단일 코드 생성만 수행                                                           |
| Repair         | execution feedback 기반 iterative correction                             |
| Planner        | planning 후 코드 생성                                                       |
| Planner-Repair | planning 이후 repair 수행                                                  |
| Proposed-v1    | failure state 및 stagnation pattern 기반 rule-driven orchestration policy |
| Proposed-v2    | replanning utilization을 강화한 aggressive orchestration policy            |

- Single-shot : 한 번의 생성만 수행하는 기본 baseline 전략
- Repair : 에러 메시지와 실패 코드를 기반으로 반복 수정하는 전략
- Planning : 문제 해결 계획을 먼저 생성한 뒤 코드를 작성하는 전략
- Planning + Repair : planning과 repair를 순차 결합한 hybrid 전략
- Rule-based Adaptive policy : 중간 상태와 실패 유형을 기반으로 전략(reasoning strategy)을 선택하는 제안 방식

### Policy Configuration Table

| Method      | max_plan_calls | max_repair_steps | stagnation_threshold | Policy Character                                               |
| ----------- | -------------: | ---------------: | -------------------: | -------------------------------------------------------------- |
| Proposed-v1 |              5 |               10 |                    2 | repair를 충분히 허용하는 보수적 replanning policy                         |
| Proposed-v2 |              8 |                4 |                    2 | repair depth를 줄이고 planning 기회를 늘린 aggressive replanning policy |


## 📁 Folder Structure
```
project/
│
├── README.md
│
├── analysis/                # notebook, 결과 분석 및 정리
│   ├── data_EDA/            # 활용할 데이터 분석을 통해 TASK 구체화
│   └── .../          
│
├── datasets/          
│   ├── humaneval/
│   │   └── raw/
│   ├── mbpp/
│   ├── humaneval_pro/
│   └── mbpp_pro/
├── src/                     # 핵심 코드
│   ├── models/              # sLM, LLM wrapper
│   │   ├── base.py        
│   │   └── hf_model.py
│   ├── adapters/            # dataset별 실행 포맷 차이 처리
│   │   ├── base.py
│   │   ├── humaneval.py
│   │   └── mbpp.py
│   ├── orchestration/       # orchestration : 전략 로직 파트
│   │   ├── policy_v1.py 
│   │   ├── policy_v2.py 
│   │   ├── code_then_plan_repair.py 
│   │   ├── code_then_plan.py 
│   │   ├── repair.py               
│   │   └── single.py
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
│   ├── phi/
│   │   ├── configs_humaneval/
│   │   └── configs_mbpp/
│   │
│   ├── qwen/
│   │   ├── configs_humaneval/
│   │   └── configs_mbpp/
│
├── results/                 
│   ├── phi35mini/
│   │   ├── humaneval/           
│   │   ├── mbpp/     
│   │   │   ├── single/
│   │   │   ├── repair/
│   │   │   │     ├── summary.json
│   │   │   │     ├── step_logs.jsonl
│   │   │   │     └── ...
│   │   │   └── .../
│   │   └── .../
│   │
│   └── qwen7bcoder/
│       └── ...
│
├── scripts/                 # 공용 스크립트
│
├── logs/                    # raw 실행 로그
│
└── README.md
```

<!--
| Method      | Description                                                                                                                                                                                         |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Proposed-v1 | failure state와 반복 실패 패턴을 기반으로 repair와 replanning을 선택하는 rule-based orchestration policy. Planning은 최대 5회까지 허용하며, 각 policy cycle에서 repair를 최대 10회까지 수행할 수 있어 local correction을 충분히 시도하는 구조이다.         |
| Proposed-v2 | Proposed-v1과 동일한 rule-based decision structure를 유지하되, planning budget을 8회로 늘리고 repair depth를 4회로 줄인 variant이다. 따라서 local repair보다 global replanning을 더 자주 수행하도록 설계된 aggressive replanning policy이다. |

Proposed-v1과 Proposed-v2는 동일한 failure-state 기반 rule structure를 공유하지만, recovery budget allocation에서 차이를 가진다. Proposed-v1은 더 긴 repair sequence를 허용하여 local correction을 충분히 수행하는 repair-heavy policy이며, Proposed-v2는 planning budget을 늘리고 repair depth를 줄여 global replanning을 더 적극적으로 수행하는 planning-heavy policy이다.
-->
