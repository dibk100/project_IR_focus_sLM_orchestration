### Model
- Qwen/Qwen2.5-Coder-7B-Instruct

### Dataset
- HumanEval (164 problems)

### Setting
- Single-shot
- temperature = 0.0
- max_new_tokens = 512

### Execution
- Local inference (transformers, no vLLM server)
- No sampling (deterministic decoding)

### Overall Performance 📊
- Total: 164
- Pass: 118
- Fail: 46
- Timeout: 0
- Pass@1: 0.7195

### Error Type Distribution 📉
- assertion_error: 28
- syntax_error: 13
- name_error: 3
- type_error: 1
- value_error: 1

### Latency ⏱
- Avg: ~2.1s
- Max: ~6.6s

### Extraction Issues (Resolved)
- Initial runs showed high syntax_error (~65 cases)
- Root cause: markdown code fences (```python) leaking into execution
- Fix: minimal sanitization (remove ``` markers)

### Design Decision
- Avoid complex extraction logic
- Preserve "continuation" assumption of HumanEval
- Use minimal postprocessing to avoid breaking indentation

### Interpretation

- After minimal sanitization, failure mode shifts from formatting issues to semantic errors.
- This indicates that the baseline is now suitable for evaluating orchestration effects.
- Remaining failures are primarily due to incorrect logic rather than invalid code generation.

### Next Step Decision

→ Move to repair loop

Rationale:
- Baseline is stable and interpretable
- Failure types are actionable (mostly assertion_error)
- Repair can meaningfully target semantic errors

### Open Questions
- How much of assertion_error can be fixed via simple repair?
- Does repair primarily fix syntax or semantic errors?
- What is the cost (latency / attempts) of repair?

### TODO
- Implement repair loop (max_attempts=2)
- Log attempt-wise results
- Compare success improvement vs single-shot