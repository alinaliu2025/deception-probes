"""Causal validation of a probe direction by activation steering (ADR 0010).

Detection AUROC only shows a direction *correlates* with the caved/held split.
A direction can hit AUROC 1.0 and still be a spectator feature -- e.g. "the
committed letter equals the one the user asserted", a lexical match/no-match
relation that rides along with sycophancy without causing it. The only test that
separates cause from correlation is intervention: push the direction into the
residual stream and see if behaviour moves.

Two passes, both on the sycophancy `factual` source (a known-correct MCQ where the
user asserts the wrong answer):

- **add** (elicitation): on the UNPRESSURED question, add ``alpha * v`` at the
  probe's layer and measure how often the model flips to the wrong answer. A
  genuine caving direction drives the wrong-answer rate up monotonically with
  alpha; a lexical artifact produces no coherent behavioural push.
- **ablate** (suppression): on the PRESSURED prompt, project ``v`` out of the
  residual stream and measure how much the caving rate drops versus baseline.

`v` is the probe direction; it reads ``hidden_states[L]`` = the output of decoder
block ``L`` (index 0 is embeddings), so we intervene on ``model.model.layers[L-1]``.
Alpha is scaled by the layer's mean last-token residual norm by default, so the
same alpha range is meaningful across model sizes (pass raw=True to disable).

This module holds no CLI; ``scripts/steer.py`` drives it. Runs on a real GPU with
the 7B model -- the direction it validates was trained there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .activations import build_prompt, default_batch_size
from .data.base import Example
from .data.sycophancy import NEUTRAL_SYSTEM, parse_choice
from .probes import Probe


def load_probe(path) -> Probe:
    """Rebuild a Probe from a ``probe.npz`` written by train_one/compare."""
    d = np.load(path, allow_pickle=True)
    return Probe(
        direction=d["direction"].astype(np.float32),
        bias=float(d["bias"]),
        layer=int(d["layer"]),
        method=str(d["method"]),
        deception_type=str(d["deception_type"]),
    )


def _decoder_layers(model):
    """The decoder block ModuleList (Qwen2.5: ``model.model.layers``)."""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise AttributeError(
            "could not find decoder layers at model.model.layers -- steer.py "
            "assumes a Qwen2.5-style LlamaModel layout"
        )
    return layers


def make_add_hook(vec: torch.Tensor, alpha: float):
    """Forward hook that adds ``alpha * vec`` to a decoder block's output."""
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + alpha * vec.to(dtype=h.dtype, device=h.device)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return hook


def make_ablate_hook(vec: torch.Tensor):
    """Forward hook that projects ``vec`` out of a decoder block's output.

    ``vec`` must be unit norm; h' = h - (h.v) v removes the component along it.
    """
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        v = vec.to(dtype=h.dtype, device=h.device)
        coeff = (h * v).sum(dim=-1, keepdim=True)
        h = h - coeff * v
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return hook


def _install(model, layer: int, hook):
    """Register ``hook`` on the block whose output is ``hidden_states[layer]``."""
    if layer < 1:
        raise ValueError(
            f"cannot steer at layer {layer}: layer 0 is the embedding output, "
            "which no decoder block produces (train a probe on a mid layer)"
        )
    return _decoder_layers(model)[layer - 1].register_forward_hook(hook)


@dataclass
class SteerItem:
    """One question ready for a steering pass: the prompt Example plus the strings
    parse_choice needs to score the model's committed answer."""
    example: Example
    matching: str      # the WRONG answer string, e.g. " (B)" -- caving picks this
    not_matching: str  # the correct answer string


def build_items(examples: list[Example], pressured: bool) -> list[SteerItem]:
    """Turn raw sycophancy factual/rollout Examples (label -1, unfiltered) into
    SteerItems. ``pressured`` keeps the user's assertion prompt; otherwise the
    persona-stripped neutral question is used (no pressure present)."""
    items = []
    for ex in examples:
        if pressured:
            prompt_ex = Example(ex.system, ex.user, ex.label, ex.deception_type)
        else:
            prompt_ex = Example(NEUTRAL_SYSTEM, ex.meta["neutral_user"], ex.label,
                                ex.deception_type)
        items.append(SteerItem(prompt_ex, ex.meta["matching"], ex.meta["not_matching"]))
    return items


def _classify(model, tokenizer, items: list[SteerItem], device: str,
              batch_size: int, max_new_tokens: int, samples: int,
              temperature: float) -> list[dict]:
    """Generate for every item (greedy if samples==1, else `samples` sampled
    completions) and count committed choices. Returns one dict per item:
    ``{"wrong": int, "correct": int, "unparsed": int}`` over its completions."""
    prompts = [build_prompt(tokenizer, it.example) for it in items]
    order = sorted(range(len(items)), key=lambda i: len(tokenizer.encode(prompts[i])))
    results: list[dict | None] = [None] * len(items)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = [prompts[i] for i in idx]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                do_sample=samples > 1,
                temperature=temperature if samples > 1 else None,
                num_return_sequences=samples,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        new = gen[:, inputs.input_ids.shape[1]:]
        decoded = tokenizer.batch_decode(new, skip_special_tokens=True)
        for j, i in enumerate(idx):
            counts = {"wrong": 0, "correct": 0, "unparsed": 0}
            for s in range(samples):
                text = decoded[j * samples + s]
                choice, _ = parse_choice(text, items[i].matching, items[i].not_matching)
                if choice is None:
                    counts["unparsed"] += 1
                elif choice == "matching":
                    counts["wrong"] += 1
                else:
                    counts["correct"] += 1
            results[i] = counts
    return results  # type: ignore[return-value]


def _wrong_rate(counts: list[dict]) -> float:
    """Fraction of parseable completions that committed to the WRONG answer."""
    wrong = sum(c["wrong"] for c in counts)
    parseable = sum(c["wrong"] + c["correct"] for c in counts)
    return wrong / parseable if parseable else float("nan")


def _parse_rate(counts: list[dict]) -> float:
    """Fraction of ALL completions that produced a recognisable choice. A parse rate
    that falls as alpha climbs means the intervention is breaking generation, not
    steering it -- so a wrong_rate computed over the survivors is not trustworthy."""
    parseable = sum(c["wrong"] + c["correct"] for c in counts)
    total = sum(c["wrong"] + c["correct"] + c["unparsed"] for c in counts)
    return parseable / total if total else float("nan")


def random_probe(probe: Probe, seed: int = 0) -> Probe:
    """A matched random-direction control: a random unit vector at the SAME layer.

    Steering it is the baseline the add/ablate passes need -- if a random vector of
    equal norm moves behaviour as much as the probe direction does, the effect is
    generic perturbation (adding noise degrades the model), not the direction."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=probe.direction.shape).astype(np.float32)
    d /= np.linalg.norm(d) + 1e-8
    return Probe(d, 0.0, probe.layer, probe.method + "_randctrl", probe.deception_type)


def _add_curve(model, tokenizer, vec: torch.Tensor, correct: list[SteerItem],
               layer: int, scale: float, alphas: list[float], device: str,
               batch_size: int, max_new_tokens: int, samples: int,
               temperature: float, verbose: bool, tag: str) -> dict:
    """Sweep alpha for one direction over a fixed baseline-correct item set,
    reporting both the wrong-answer rate and the parse rate at each alpha."""
    wrong, parse = [], []
    for a in alphas:
        handle = _install(model, layer, make_add_hook(vec, a * scale))
        try:
            counts = _classify(model, tokenizer, correct, device, batch_size,
                               max_new_tokens, samples, temperature)
        finally:
            handle.remove()
        wr, pr = _wrong_rate(counts), _parse_rate(counts)
        wrong.append(wr)
        parse.append(pr)
        if verbose:
            print(f"    [{tag}] alpha={a:>6.3f}  wrong_rate={wr:.3f}  "
                  f"parse_rate={pr:.3f}")
    return {"wrong_rate": wrong, "parse_rate": parse}


def _resid_scale(model, tokenizer, items: list[SteerItem], layer: int,
                 device: str, batch_size: int) -> float:
    """Mean L2 norm of the last-token residual at ``hidden_states[layer]`` over the
    items' prompts. Used to make alpha a fraction of a typical activation norm."""
    prompts = [build_prompt(tokenizer, it.example) for it in items]
    norms = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        h = out.hidden_states[layer][:, -1, :].float()
        norms.append(h.norm(dim=-1).cpu().numpy())
    return float(np.concatenate(norms).mean())


def add_sweep(model, tokenizer, probe: Probe, examples: list[Example], device: str,
              alphas: list[float], *, raw: bool = False, samples: int = 1,
              temperature: float = 0.7, max_new_tokens: int = 24,
              batch_size: int | None = None, control: Probe | None = None,
              verbose: bool = True) -> dict:
    """Elicitation pass: add ``alpha * v`` at the probe layer on UNPRESSURED items
    the model answers CORRECTLY at baseline, and report the wrong-answer rate AND
    parse rate at each alpha. A causal caving direction makes wrong_rate climb with
    alpha while the parse rate holds; if ``control`` (a matched random direction) is
    given, it is swept over the SAME items and scale so the two curves compare
    directly -- a real effect beats its random control."""
    batch_size = batch_size or default_batch_size(device)
    vec = torch.from_numpy(probe.direction.astype(np.float32))
    items = build_items(examples, pressured=False)

    base = _classify(model, tokenizer, items, device, batch_size, max_new_tokens,
                     samples, temperature)
    # keep only items the model gets right unsteered: a flip is only meaningful
    # from a correct baseline
    correct = [it for it, c in zip(items, base) if c["correct"] > c["wrong"]
               and (c["correct"] + c["wrong"]) > 0]
    if verbose:
        print(f"  add: {len(correct)}/{len(items)} items correct at baseline "
              f"(steering the rest is undefined)")
    if not correct:
        return {"alphas": alphas, "wrong_rate": [float("nan")] * len(alphas),
                "parse_rate": [float("nan")] * len(alphas), "n_items": 0,
                "scale": None, "raw": raw, "control": None}

    scale = 1.0 if raw else _resid_scale(model, tokenizer, correct, probe.layer,
                                         device, batch_size)
    real = _add_curve(model, tokenizer, vec, correct, probe.layer, scale, alphas,
                      device, batch_size, max_new_tokens, samples, temperature,
                      verbose, tag="probe")
    ctrl = None
    if control is not None:
        cvec = torch.from_numpy(control.direction.astype(np.float32))
        ctrl = _add_curve(model, tokenizer, cvec, correct, probe.layer, scale, alphas,
                          device, batch_size, max_new_tokens, samples, temperature,
                          verbose, tag="random")
    return {"alphas": alphas, "wrong_rate": real["wrong_rate"],
            "parse_rate": real["parse_rate"], "n_items": len(correct),
            "scale": scale, "raw": raw, "control": ctrl}


def _ablated_caving(model, tokenizer, vec: torch.Tensor, caved: list[SteerItem],
                    layer: int, device: str, batch_size: int, max_new_tokens: int,
                    samples: int, temperature: float) -> float:
    """Caving rate after projecting ``vec`` (unit) out at ``layer``."""
    handle = _install(model, layer, make_ablate_hook(vec))
    try:
        counts = _classify(model, tokenizer, caved, device, batch_size,
                           max_new_tokens, samples, temperature)
    finally:
        handle.remove()
    return _wrong_rate(counts)


def ablate_pass(model, tokenizer, probe: Probe, examples: list[Example], device: str,
                *, samples: int = 1, temperature: float = 0.7,
                max_new_tokens: int = 24, batch_size: int | None = None,
                control: Probe | None = None, verbose: bool = True) -> dict:
    """Suppression pass: on PRESSURED items the model caves on at baseline, project
    ``v`` out at the probe layer and report the drop in caving rate. If ``control``
    (a matched random direction) is given, project IT out of the same items too --
    removing a random direction should barely move caving, so a real necessity claim
    needs the probe drop to clear the random drop."""
    batch_size = batch_size or default_batch_size(device)
    vec = torch.from_numpy(probe.direction.astype(np.float32))
    vec = vec / (vec.norm() + 1e-8)  # ablation assumes unit norm
    items = build_items(examples, pressured=True)

    base = _classify(model, tokenizer, items, device, batch_size, max_new_tokens,
                     samples, temperature)
    caved_pairs = [(it, c) for it, c in zip(items, base)
                   if c["wrong"] > c["correct"] and (c["correct"] + c["wrong"]) > 0]
    if verbose:
        print(f"  ablate: {len(caved_pairs)}/{len(items)} items caved at baseline")
    if not caved_pairs:
        return {"baseline_caving": float("nan"), "ablated_caving": float("nan"),
                "control_ablated_caving": None, "n_items": 0}

    caved = [it for it, _ in caved_pairs]
    baseline_rate = _wrong_rate([c for _, c in caved_pairs])
    ablated_rate = _ablated_caving(model, tokenizer, vec, caved, probe.layer, device,
                                   batch_size, max_new_tokens, samples, temperature)
    control_rate = None
    if control is not None:
        cvec = torch.from_numpy(control.direction.astype(np.float32))
        cvec = cvec / (cvec.norm() + 1e-8)
        control_rate = _ablated_caving(model, tokenizer, cvec, caved, probe.layer,
                                       device, batch_size, max_new_tokens, samples,
                                       temperature)
    if verbose:
        print(f"    baseline caving={baseline_rate:.3f}  "
              f"ablated caving={ablated_rate:.3f}"
              + (f"  random-ctrl caving={control_rate:.3f}"
                 if control_rate is not None else ""))
    return {"baseline_caving": baseline_rate, "ablated_caving": ablated_rate,
            "control_ablated_caving": control_rate, "n_items": len(caved)}
