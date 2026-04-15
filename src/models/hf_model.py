"""
HuggingFace 모델 Wrapper
transformers 라이브러리를 사용하여 sLM을 로드하고 추론한다.
"""
from __future__ import annotations

import gc
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.base import BaseModel


class HFModel(BaseModel):
    """HuggingFace 모델 wrapper"""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"🔄 모델 로딩: {model_name} (device={self.device})")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # pad_token이 없는 모델 대응
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device == "cuda" else None,
        )

        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self.model.eval()
        print("✅ 모델 로딩 완료")

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        프롬프트를 입력받아 생성 결과를 구조화하여 반환한다.

        Returns:
            {
                "text": str,             # 생성된 텍스트(입력 제외)
                "input_tokens": int,     # 프롬프트 토큰 수
                "output_tokens": int,    # 생성 토큰 수
                "total_tokens": int,     # input + output
            }
        """
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", self.temperature)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = int(inputs["input_ids"].shape[1])

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95
            gen_kwargs["top_k"] = 50
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # 입력 이후의 생성 구간만 분리
        generated_tokens = outputs[0][input_len:]
        output_len = int(generated_tokens.shape[0])

        generated_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        result = {
            "text": generated_text,
            "input_tokens": input_len,
            "output_tokens": output_len,
            "total_tokens": input_len + output_len,
        }

        # CPU RAM 누수 방지: 텐서 명시적 해제
        del outputs, generated_tokens, inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def get_model_name(self) -> str:
        return self.model_name