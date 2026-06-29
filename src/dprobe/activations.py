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


def _pick_dtype(device: str):
    """Pick the fastest safe dtype for the device.

    Pascal consumer GPUs (compute 6.x, e.g. GTX 1070 Ti) run fp16 at ~1/64 the
    fp32 rate -> fp16 there is catastrophically slow. Only use fp16 on Volta+
    (compute >= 7.0) and bf16 on Ampere+ (compute >= 8.0). Everything else fp32.
    """
    if device != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        return torch.bfloat16
    if major >= 7:
        return torch.float16
    return torch.float32


def default_batch_size(device: str) -> int:
    """Pick a batch size from the actual card's VRAM.

    Keyed on detected memory, not a cluster flag, so it can't desync from the
    node you're really on. Tuned for extraction (one no_grad forward, all hidden
    states materialised): 80GB A100/H100 -> 64, 32GB V100 -> 32, smaller -> 8.
    Override with --batch-size if a long sequence length still OOMs.
    """
    if device != "cuda":
        return 16
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_gb > 60:
        return 64
    if total_gb > 24:
        return 32
    return 8


def load_model(name: str = MODEL_NAME, device: str | None = None):
    """Returns (model, tokenizer, device). Call once, reuse for every dataset."""
    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=_pick_dtype(device)
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


def extract(model, tokenizer, examples: list[Example], device: str,
            verbose: bool = True, batch_size: int | None = None,
            acts_dtype: np.dtype = np.float32):
    """Run every example through the model in batches.

    Examples are sorted by prompt length before batching to minimise padding
    waste; results are reordered to match the original input order.

    batch_size defaults to a VRAM-aware value (see default_batch_size); pass an
    int to override.

    `acts_dtype` is the dtype of the returned activation buffer, the run's
    dominant host-RAM cost ([n_examples, n_layers+1, hidden]). float16 halves it
    and is lossless when the model itself runs in fp16 (the hidden states are
    already fp16; we only upcast them to compute -- see the .float() below). On a
    bf16 model fp16 can clip outlier activations, so keep the float32 default
    unless you know the node is fp16. The layer sweep upcasts each slice for
    fitting, so probe math is unaffected by the storage dtype either way.

    Left-padding means every sequence's last real token sits at position -1,
    so we extract h[:, -1, :] without tracking per-example lengths.

    Returns:
        acts:   float array [n_examples, n_layers+1, hidden] (dtype=acts_dtype)
        labels: int array   [n_examples]   (1 = deceptive condition, 0 = control)
    """
    if batch_size is None:
        batch_size = default_batch_size(device)
    if verbose:
        print(f"  batch_size={batch_size}")
    prompts = [build_prompt(tokenizer, ex) for ex in examples]
    lengths = [len(tokenizer.encode(p)) for p in prompts]
    order = sorted(range(len(examples)), key=lambda i: lengths[i])

    # labels are known up front and stay in input order
    labels = np.array([ex.label for ex in examples])
    acts = None  # preallocated on the first batch, once n_layers+1 and hidden are known
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch_prompts = [prompts[i] for i in idx]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        # Slice last token on GPU, stack all layers, ONE transfer per batch:
        # [n_layers+1, batch, hidden] -> [batch, n_layers+1, hidden]
        last = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=0)
        last = last.permute(1, 0, 2).float().cpu().numpy()
        if acts is None:
            # [n_examples, n_layers+1, hidden]; fill in place so we never hold a
            # list-of-arrays AND an np.stack copy at once (that was the 2x peak)
            acts = np.empty((len(examples), last.shape[1], last.shape[2]), dtype=acts_dtype)
        for j, i in enumerate(idx):
            acts[i] = last[j]  # write straight into the original-input slot
        if verbose:
            done = min(start + batch_size, len(examples))
            print(f"  {done}/{len(examples)}", end="\r", flush=True)
    if verbose:
        print()

    return acts, labels
