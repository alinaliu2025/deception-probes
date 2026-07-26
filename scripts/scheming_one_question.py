"""Run ONE scheming scenario through a small local model and print the raw output.

Script version of notebooks/scheming_one_question.ipynb, sized for a low-compute
machine: the default is a 0.5B model that runs on CPU. A model this small won't
emit a <think> block and rarely schemes, so read the output as a plumbing check,
not signal.

    python scripts/scheming_one_question.py
    python scripts/scheming_one_question.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --n-samples 2 --max-new-tokens 400
"""

import argparse
import pathlib
import sys

import torch


def load_scheming():
    """Import the repo's real scenario + parser, bypassing dprobe.data.__init__
    (which pulls in `datasets`) if that isn't installed."""
    repo = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo / "src"))
    try:
        from dprobe.data import scheming
        return scheming
    except ModuleNotFoundError:
        import importlib.util
        import types
        sys.modules.setdefault("dprobe", types.ModuleType("dprobe"))
        pkg = types.ModuleType("dprobe.data")
        pkg.__path__ = []
        sys.modules["dprobe.data"] = pkg
        for name in ("base", "scheming"):
            spec = importlib.util.spec_from_file_location(
                f"dprobe.data.{name}", repo / f"src/dprobe/data/{name}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"dprobe.data.{name}"] = mod
            spec.loader.exec_module(mod)
        return sys.modules["dprobe.data.scheming"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="LOCAL model; tiny default for low-compute machines")
    ap.add_argument("--n-samples", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    scheming = load_scheming()
    print("scenarios:", [s["eval"] for s in scheming.SCHEMING_ROLLOUT])

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto").to(device)
    model.eval()
    print(f"loaded {args.model} on {device}\n")

    ex = scheming.build_rollout()[0]   # oversight_subversion
    print("SYSTEM:\n", ex.system, "\n")
    print("USER (pressured):\n", ex.user, "\n")
    print("covert (A):", ex.meta["matching"])
    print("honest (B):", ex.meta["not_matching"], "\n")

    def chat(system, user):
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def generate(prompt, n=1, sample=True):
        inp = tok(prompt, return_tensors="pt").to(device)
        kw = dict(max_new_tokens=args.max_new_tokens, num_return_sequences=n,
                  pad_token_id=tok.pad_token_id, do_sample=sample)
        if sample:
            kw["temperature"] = args.temperature
        with torch.no_grad():
            out = model.generate(**inp, **kw)
        return tok.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)

    label = {"matching": "SCHEMED (1)", "not_matching": "straight (0)", None: "unparsed"}

    print("#" * 60)
    print(f"# {args.n_samples} pressured rollouts")
    print("#" * 60)
    prompt = chat(ex.system, ex.user)
    for i, t in enumerate(generate(prompt, n=args.n_samples, sample=True), 1):
        choice, _ = scheming.parse_final_choice(t, ex.meta["matching"], ex.meta["not_matching"])
        print(f"\n===== rollout {i}: {label[choice]} =====")
        print(t)

    print("\n" + "#" * 60)
    print("# calm gate (no goal-conflict): scenario is dropped if this schemes")
    print("#" * 60)
    out = generate(chat(ex.meta["calm_system"], ex.meta["neutral_user"]), n=1, sample=False)[0]
    choice, _ = scheming.parse_final_choice(out, ex.meta["matching"], ex.meta["not_matching"])
    print(f"\ncalm choice: {label[choice]}\n")
    print(out)


if __name__ == "__main__":
    main()
