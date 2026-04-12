from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ptq.config import ModelSettings


def load_model_and_tokenizer(
    model_settings: ModelSettings,
    device: str,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_settings.model_id,
        trust_remote_code=model_settings.trust_remote_code,
    )

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_settings.model_id,
        trust_remote_code=model_settings.trust_remote_code,
        torch_dtype=dtype,
    )
    model.eval()
    model.to(device)
    return model, tokenizer
