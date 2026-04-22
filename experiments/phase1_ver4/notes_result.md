# 📘 Phase 1: Orchestration Method Comparison & Analysis (HumanEval, MBPP, BigCodeBench)
## ver2 Results_hummaneval

Base
```
  📊 결과 요약
  총 문제: 164
  통과: 113
  실행 성공: 140
  pass@1: 0.6890
  execution_success_rate: 0.8537
  conditional_pass: 0.8071
============================================================
  code_failed: 24
  define_test_failed: 0
  run_test_failed: 27
```

Repair
```
  📊 결과 요약
  총 문제: 164
  통과: 141
  실행 성공: 144
  pass@1: 0.8598
  execution_success_rate: 0.8780
  conditional_pass: 0.9792
============================================================
  code_failed: 20
  define_test_failed: 0
  run_test_failed: 3
```

code-then-plan
```
📊 결과 요약
  총 문제: 164
  통과: 147
  실행 성공: 161
  pass@1: 0.8963
  execution_success_rate: 0.9817
  conditional_pass: 0.9130
============================================================
  code_failed: 3
  define_test_failed: 0
  run_test_failed: 14
============================================================
  plan 사용: 49/164 (29.9%)
  plan 복구 성공: 32/49 (65.3%)
```

ours
```
📊 결과 요약
  총 문제: 164
  통과: 141
  실행 성공: 152
  pass@1: 0.8598
  execution_success_rate: 0.9268
  conditional_pass: 0.9276
```