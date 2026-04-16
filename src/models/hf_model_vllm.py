# src/models/hf_model.py
import time
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class HFModel:
    model_name: str
    max_new_tokens: int = 512
    temperature: float = 0.2
    backend: str = "hf"   # "hf" | "vllm_server"
    api_base: str = None
    api_key: str = "EMPTY"

    def __post_init__(self):
        if self.backend == "vllm_server":
            if not self.api_base:
                raise ValueError("api_base is required for vllm_server backend")
            self.client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
            )
        else:
            # 기존 HF 로컬 로딩 코드 유지
            self.client = None
            self._init_hf_model()

    def _init_hf_model(self):
        # 기존 transformers 로딩 코드
        pass

    def generate(self, prompt: str) -> dict:
        if self.backend == "vllm_server":
            return self._generate_vllm(prompt)
        return self._generate_hf(prompt)

    def _generate_vllm(self, prompt: str) -> dict:
        t0 = time.perf_counter()

        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

        latency = time.perf_counter() - t0
        text = response.choices[0].text

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) if usage else (prompt_tokens + completion_tokens)

        return {
            "text": text,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_sec": latency,
            "backend": "vllm_server",
            "model_name": self.model_name,
        }

    def _generate_hf(self, prompt: str) -> dict:
        # 기존 HF generate 로직
        raise NotImplementedError