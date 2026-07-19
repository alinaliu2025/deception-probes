"""Load a model and pull last-token residual-stream activations.

This is the part the LiteLLM proxy can't do for you: probes need the residual
stream, which only a locally loaded model exposes. Ported almost verbatim from
the original train_deception_probe.py so the behaviour is identical.
"""

from __future__ import annotations

import re

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
        name, dtype=_pick_dtype(device)
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
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    # rollout design (ADR 0008): an in-progress assistant answer, truncated at
    # the choice-commit token, is appended UNCLOSED (no end-of-turn) so the
    # last token -- the extract() read position -- is the commit token itself.
    if "assistant_prefix" in ex.meta:
        prompt += ex.meta["assistant_prefix"]
    return prompt


def verify_read_positions(tokenizer, examples: list[Example], n: int = 3,
                          verbose: bool = True) -> None:
    """Index-verification guard (v2-plan guard 2): show, and where possible assert,
    the token that ``extract`` reads the hidden state from.

    ``extract`` reads ``h[:, -1, :]`` -- the LAST token of the prompt. For a
    rollout example that token is meant to be the answer-commit token (the
    ``(A)``/``(B)``). This is the most common silent bug: an off-by-one read lands
    on a space / paren / end-of-turn instead, quietly poisoning the direction.

    For the first ``n`` examples this decodes the final tokens (so you can eyeball
    the read position) and, when the example carries a rollout ``assistant_prefix``
    with a letter, ASSERTS that letter is among those final tokens -- raising
    before the expensive extraction pass rather than after a wasted run.
    """
    if not examples:
        return
    if verbose:
        print(f"read-position check (extract reads the LAST token; first "
              f"{min(n, len(examples))} examples):")
    for ex in examples[:n]:
        prompt = build_prompt(tokenizer, ex)
        ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
        tail = ids[-6:]
        toks = [tokenizer.decode(t) for t in tail]
        if verbose:
            print(f"  label={ex.label} | last 6 tokens: "
                  f"{' '.join(repr(t) for t in toks)}")
            print(f"    >>> READ POSITION (last token) = "
                  f"{tokenizer.decode(ids[-1:])!r}")
        prefix = ex.meta.get("assistant_prefix")
        if prefix is not None:
            m = re.search(r"\(([A-Z])\)", prefix)
            if m:
                letter = m.group(1)
                tail_text = tokenizer.decode(ids[-4:])
                assert letter in tail_text, (
                    "read-position guard FAILED: expected the answer letter "
                    f"({letter}) among the final tokens, but the last tokens "
                    f"decode to {tail_text!r}. The hidden state would be read "
                    "off the wrong token -- fix the prefix/extraction before "
                    "trusting this run."
                )
    if verbose:
        print("  read-position check passed")


def seq_logprob(model, tokenizer, prompt: str, continuation: str, device: str) -> float:
    """Total log-prob the model assigns to `continuation` following `prompt`.

    Teacher-forced: one forward pass over prompt+continuation, summing the log-prob
    of each continuation token. Used by sycophancy.behavior_filter to decide which
    of two MCQ answers the model prefers (score each, pick the higher) without
    sampling. The shared-prefix length is found by token-id match so a tokenizer
    boundary merge between prompt and continuation can't shift the scored span.
    Returns -inf if the continuation adds no tokens.
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    full_ids = tokenizer(prompt + continuation, return_tensors="pt").input_ids[0]
    n = 0
    while (n < len(prompt_ids) and n < len(full_ids)
           and int(prompt_ids[n]) == int(full_ids[n])):
        n += 1
    if n >= len(full_ids):
        return float("-inf")
    inp = full_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp).logits[0].float()  # [T, vocab]
    logprobs = torch.log_softmax(logits, dim=-1)
    # token at position i is predicted by the logits at position i-1
    total = 0.0
    for i in range(n, len(full_ids)):
        total += float(logprobs[i - 1, int(full_ids[i])])
    return total


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
