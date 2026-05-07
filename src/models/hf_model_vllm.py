"""
HuggingFace 모델 Wrapper (HF local + vllm_server 통합)

backend="hf"          : transformers 로컬 추론 (기존 동작)
backend="vllm_server" : vLLM OpenAI-compatible server 추론
"""
import gc
import time
from typing import Any, Dict, Optional

import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.base import BaseModel
import httpx



class HFModel(BaseModel):

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        backend: str = "hf",          # "hf" | "vllm_server"
        api_base: Optional[str] = None,
        api_key: str = "EMPTY",
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.backend = backend

        if self.backend == "vllm_server":
            if not api_base:
                raise ValueError("api_base is required for vllm_server backend")
            
            timeout = httpx.Timeout(
                connect=10.0,
                read=120.0,
                write=10.0,
                pool=10.0,
            )
            self.client = OpenAI(
                base_url=api_base,
                api_key=api_key,
                timeout=timeout,
            )
                        
            # HF 관련 속성은 사용하지 않음
            self.model = None
            self.tokenizer = None
            self.device = None
            print(f"✅ vllm_server 연결: {api_base} | model={model_name}")

        else:
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device

            print(f"🔄 모델 로딩: {model_name} (device={self.device})")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
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
            self.client = None
            print("✅ 모델 로딩 완료")

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        if self.backend == "vllm_server":
            return self._generate_vllm(prompt, **kwargs)
        return self._generate_hf(prompt, **kwargs)

    def _generate_vllm(self, prompt: str, **kwargs) -> Dict[str, Any]:
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", self.temperature)

        t0 = time.perf_counter()
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        # response = self.client.chat.completions.create(
        #     model=self.model_name,
        #     messages=[
        #         {"role": "user", "content": prompt}
        #     ],
        #     max_tokens=max_new_tokens,
        #     temperature=temperature,
        # )
        latency = time.perf_counter() - t0

        text = response.choices[0].text
        # text = response.choices[0].message.content
        usage = getattr(response, "usage", None)
        prompt_tokens     = getattr(usage, "prompt_tokens",     0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        total_tokens      = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) if usage else (prompt_tokens + completion_tokens)

        return {
            "text":         text,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_sec":  latency,
        }

    def _generate_hf(self, prompt: str, **kwargs) -> Dict[str, Any]:
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

        generated_tokens = outputs[0][input_len:]
        output_len = int(generated_tokens.shape[0])
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        del outputs, generated_tokens, inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "text":         generated_text,
            "input_tokens": input_len,
            "output_tokens": output_len,
            "total_tokens": input_len + output_len,
        }

    def get_model_name(self) -> str:
        return self.model_name