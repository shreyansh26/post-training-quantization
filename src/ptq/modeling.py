import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ptq.config import ModelSettings


def load_model_and_tokenizer(
    model_settings: ModelSettings,
    device: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the requested causal LM and move it onto the selected device."""
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
