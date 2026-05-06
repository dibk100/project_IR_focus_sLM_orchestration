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
- **Base Model**: Qwen2.5-Coder-7B-Instruct
  - microsoft/Phi-3.5-mini-instruct
  - meta-llama/Llama-3.1-8B-Instruct
- **Benchmarks**
  - HumanEval(164)
  - MBPP(374)
  - Bigcode(1140)
- **Task Type**: Code Generation, Iterative Refinement
- **Inference Budget**: Fixed call budget of 20
- **Evaluation Metric**: Pass rate, ...

## 🧩 Compared Strategies
- 1. Single-shot : 한 번의 생성만 수행하는 기본 baseline 전략
- 2. Repair : 에러 메시지와 실패 코드를 기반으로 반복 수정하는 전략
- 3. Planning : 문제 해결 계획을 먼저 생성한 뒤 코드를 작성하는 전략
- 4. Planning + Repair : planning과 repair를 순차 결합한 hybrid 전략
- 5. Rule-based Adaptive policy : 중간 상태와 실패 유형을 기반으로 전략(reasoning strategy)을 선택하는 제안 방식


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

