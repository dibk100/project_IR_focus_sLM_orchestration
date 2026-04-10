# 📘 Phase 1: Easy Code Generation (HumanEval, MBPP)
## Architecture 
```
TaskLoader -> Problem Iterator -> Strategy.run(problem, model) -> Execution/Verification -> Metrics -> Save JSONL
```
## 1. 실험 목적
- HumanEval, MBPP 통합 pipeline 구현
- input TASK 명확히 하는 것이 목표
   - HumanEval의 경우, input에 고정된 템플릿으로 넣어서 함수생성하게 하는 쉬운 task
   - MBPP의 경우, input에 자연어를 넣어서 프로그램(py)가 생성되게 하는 좀 더 어려운 task
      - **해당 부분에 LM의 오류가 많이 발생할텐데 이 오류유형을 확인하고, 각 single과 다른 오케스트레이션으로 변화가 있는지 확인하고자 함**

## 2. 실험 설정
### Datasets
- HumanEval (OpenAI, 2021) : **164 problems**
- MBPP (Google Research, 2021) : **374 problems**
### Base Model
- Qwen/Qwen2.5-Coder-7B-Instruct
    - baseline = Single-Shot
    - temperature = 0.0
    - max_new_tokens = 512

### 평가 지표
- pass@1 (= overall pass rate)
- execution success rate (= structural success)
- conditional pass (= semantic success among executable samples)
- gain vs single


## 3. Experiment Log 📊
- results/phase1_ver2/sample
   - 5개 샘플로 확인하는 작업 진행함.
   - ISSUE : mbpp의 prompting 수정이 필요함. analysis/exp_phase1_ver2에 샘플 분석 자료 올려둠. 모델이 필요없는 자연어를 출력해서 생기는 문제