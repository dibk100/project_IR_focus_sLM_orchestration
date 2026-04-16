# 📘 Phase 2: Hard Code Generation (BigCodeBench,LiveCodeBench,ClassEval)

## 1. 실험 목적
@@@

## 2. 실험 설정
### Datasets
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

