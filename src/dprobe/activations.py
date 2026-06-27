"""Load a model and pull last-token residual-stream activations.

This is the part the LiteLLM proxy can't do for you: probes need the residual
stream, which only a locally loaded model exposes. Ported almost verbatim from
the original train_deception_probe.py so the behaviour is identical.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_NAME
from .data.base import Example


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(name: str = MODEL_NAME, device: str | None = None):
    """Returns (model, tokenizer, device). Call once, reuse for every dataset."""
    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float32 if device != "cuda" else torch.float16
    ).to(device)
    model.eval()
    return model, tokenizer, device


def build_prompt(tokenizer, ex: Example) -> str:
    messages = [
        {"role": "system", "content": ex.system},
        {"role": "user", "content": ex.user},
    ]
    if ex.completion is not None:
        messages.append({"role": "assistant", "content": ex.completion})
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


@torch.no_grad()
def _activations_for_prompt(model, tokenizer, prompt: str, device: str) -> np.ndarray:
    """Last-token residual stream for every layer -> array [n_layers+1, hidden]."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**inputs, output_hidden_states=True)
    hs = torch.stack([h[0, -1, :].float().cpu() for h in out.hidden_states])
    return hs.numpy()


def extract(model, tokenizer, examples: list[Example], device: str, verbose: bool = True):
    """Run every example through the model.

    Returns:
        acts:   float array [n_examples, n_layers+1, hidden]
        labels: int array   [n_examples]   (1 = deceptive condition, 0 = control)
    """
    acts, labels = [], []
    for i, ex in enumerate(examples, 1):
        prompt = build_prompt(tokenizer, ex)
        acts.append(_activations_for_prompt(model, tokenizer, prompt, device))
        labels.append(ex.label)
        if verbose:
            print(f"  {i}/{len(examples)}", end="\r", flush=True)
    if verbose:
        print()
    return np.stack(acts), np.array(labels)
