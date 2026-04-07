# project_IR_focus_sLM_orchestration 🚀
- **Type**: 개인 연구 프로젝트 (Independent Research)
- **Area** : sLM orchestration

## Overview 📝
본 레포지토리는 소형 언어 모델(sLM)의 성능을 향상시키기 위한 오케스트레이션 전략을 연구한다.
대형 언어 모델이 모델 크기(scale)에 의존하는 것과 달리, 본 연구는 시스템 수준의 orchestration 설계가 작은 모델의 성능을 얼마나 끌어올릴 수 있는지를 탐구한다.

## Objective
sLM에서 어떤 오케스트레이션 구조가 성능 향상에 효과적인지 규명하는 것

## Research Questions
RQ1: sLM에 orchestration을 적용하면, 단일 호출(single-call) 대비 sLM 성능을 얼마나 향상시키는가?    
RQ2: 어떤 오케스트레이션 primitive가 sLM에서 가장 효과적인 성능 향상을 제공하는가?    
RQ3: 오케스트레이션을 적용한 sLM은 더 큰 모델(single-shot)과 비교했을 때 어느 수준까지 성능 격차를 줄일 수 있는가?

## Methodology
- Task: 코드 생성 및 수정 (Code Generation / Editing)
    - HumanEval: **기본 작동 여부 확인용**
    - HumanEval Pro: **난이도 상승 시 robustness 확인용**
    - (고민)SWE-bench-lite subset: **확장 가능성 점검용**
- Task 선정 이유: 코드 생성 태스크는 결과를 정량적으로 평가할 수 있음,오케스트레이션의 효과를 안정적으로 측정하기에 적합한 테스트베드로 판단함.
- 접근 방식:
    - 다양한 orchestration 구조 설계 및 비교
    - single-call baseline과 성능 비교
    - orchestration 구성 요소별 기여도 분석

## 🧪 Experimental Design
- Phase 1 : Easy Code Generation (HumanEval)
    - 목적: sLM 기반 코드 생성에서 orchestration primitive의 효과 검증
    - Dataset: HumanEval (164 problems)
    - 비교 대상:
        1. Single-shot (baseline, G)
        2. Refinement-only (G + R)
        3. Verification-only (G + V)
        4. Repair loop (G + V + R)
        5. Planner-Coder (D → G)
    - 주요 평가 지표: Pass@1, Latency, Error Type, Gain vs Single
    - 연구 질문 (Phase 1 RQ):
        - 단순 single-shot generation 대비 orchestration은 어떤 방식으로 성능을 개선하는가?
        - 각 primitive(G, R, V, D)는 성능 향상에 어떤 기여를 하며, 어떤 한계를 가지는가?
        - primitive 간 결합(R + V)은 성능에 어떤 영향을 미치는가?
        - imperfect verification signal이 refinement와 결합될 때 성능 향상을 어떻게 유도하는가?

- Phase 2 : Harder Tasks / Structure Comparison
    - 목적: 난이도가 높은 태스크에서 Phase 1에서 확인된 primitive 및 orchestration 구조의 효력 검증
    - Dataset: MBPP 또는 HumanEval Pro
    - 비교 대상: Phase 1과 동일, 필요 시 hybrid orchestration 추가
    - 연구 질문 (Phase 2 RQ):
        - 난이도가 높은 태스크에서 Phase 1에서 확인된 primitive 및 구조가 얼마나 일반화되는가?
        - 구조별 성능 차이는 난이도에 따라 어떻게 달라지는가?
        - Hybrid orchestration 조합은 기존 구조 대비 성능 향상에 기여하는가?

- Phase 3 : Scaling / sLM vs LLM
    - 목적: Phase 1~2에서 검증된 orchestration 구조가 더 큰 모델과 비교했을 때 성능 격차를 얼마나 줄이는지 평가
    - Dataset: Phase 2 동일, 필요 시 추가 benchmark
    - 비교 대상: sLM orchestration vs LLM single-shot
    - 연구 질문 (Phase 3 RQ):
        - sLM + orchestration은 LLM 단일 호출 대비 어느 수준까지 성능 격차를 줄일 수 있는가?
        - 비용(연산량, latency 등) 대비 성능 개선 효과는 어느 정도인가?

## ⭐ Research Lineage
- v1: project_IR_explore → 아이디어 탐색 및 초기 구현
- v2: project_IR_develop → 시스템 설계 및 정책 구조 개발
- v3 (현재): 본 레포지토리 → 오케스트레이션 효과에 대한 정량적 연구

## 📁 Folder Structure
```
project/
│
├── README.md
├── configs/                 # 실험 설정 (재현성 핵심)
│
├── datasets/          
│   ├── humaneval/
│   │   ├── raw/
│   │   └── processed/
│   ├── mbpp/
│   ├── humaneval_pro/
│   └── swebench_lite/
├── src/                     # 핵심 코드
│   ├── models/              # sLM, LLM wrapper
│   ├── orchestration/       # planner, repair loop 등
│   ├── tasks/               # task interface (HumanEval 등)
│   └── evaluation/          # metric 계산
│
├── experiments/
│   ├── phase1_easy/
│   │   ├── configs/
│   │   ├── run.sh
│   │   └── notes.md
│   │
│   ├── phase2_reasoning/
│   │   ├── configs/
│   │   ├── run.sh
│   │   └── notes.md
│   │
│   ├── phase3_repo/
│   │   ├── configs/
│   │   ├── run.sh
│   │   └── notes.md
│
├── results/                 # 가장 중요
│   ├── phase1_easy/
│   │   ├── single/
│   │   ├── repair/
│   │   ├── planner/
│   │   └── llm/
│   │
│   ├── phase2_reasoning/
│   │   └── ...
│   │
│   ├── phase3_repo/
│       └── ...
│
├── logs/                    # raw 실행 로그
│
└── analysis/                # 그래프, notebook, 결과 정리
```

