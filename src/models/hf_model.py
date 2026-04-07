"""
HuggingFace 모델 Wrapper
transformers 라이브러리를 사용하여 sLM을 로드하고 추론한다.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.models.base import BaseModel


class HFModel(BaseModel):
    """HuggingFace 모델 wrapper"""

    def __init__(
        self,
        model_name: str,
        device: str = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"🔄 모델 로딩: {model_name} (device={self.device})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        print(f"✅ 모델 로딩 완료")

    def generate(self, prompt: str, **kwargs) -> str:
        """프롬프트를 입력받아 생성된 텍스트(완성 부분만)를 반환"""
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", self.temperature)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # 입력 부분을 제외한 생성된 텍스트만 디코딩
        generated_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def get_model_name(self) -> str:
        return self.model_name
