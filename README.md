# deception-probes

Single-turn deception probes for three behaviours: **sycophancy**, **sandbagging**,
and **omission/concealment**. We train a linear probe on the residual stream of a
small local LLM for each behaviour, then compare them, the main question being
whether a probe trained on one type of deception detects another.

This is research code for Dr. Zhu's lab. It started as one script
(`train_deception_probe.py`) that did honest-vs-deceptive probing; this repo
generalises that into a small package you can extend per deception type.

---

## What a "probe" is here (the 60-second version)

1. We build **contrast pairs**: the same task in two conditions, a control (honest)
   one and a deceptive one. The only difference between them is whatever induces the
   behaviour (a system prompt, or how the user frames things).
2. We run each prompt through the model and grab the **last-token residual-stream
   activation at every layer**.
3. We fit a **linear probe** on those activations: a single direction in activation
   space that separates control from deceptive. Two methods:
   - `mms` = difference of means (`mean(deceptive) - mean(honest)`)
   - `lr` = logistic regression (the weight vector is the direction)
4. We **sweep every layer**, pick the best one by held-out AUROC, and report.

The proxy (LiteLLM) can't do this. Probes need the residual stream, which only a
locally loaded model exposes. So everything here runs a model locally (tiny one on a
laptop, bigger one on OSC when you want real numbers).

---

## The actual research question

Per type we get a probe and an AUROC. The interesting part is the comparison:

- **Transfer matrix** (`scripts/compare.py`): train a probe on type A, evaluate it on
  type B, for all pairs. A strong off-diagonal means the three behaviours live on a
  shared "deception" direction. A weak one means they're distinct and you'd need a
  type-specific probe for each. This is the single-turn analogue of the transfer
  question in the long-horizon work.
- **Direction geometry**: cosine similarity between the three learned directions.
  ~1 = same axis, ~0 = orthogonal. A sanity check on the transfer matrix.

---

## Repo layout

```
deception-probes/
├── README.md
├── CLAUDE.md                 # context for Claude Code (loaded every session)
├── pyproject.toml            # makes `pip install -e .` work
├── requirements.txt
├── src/dprobe/
│   ├── config.py             # model name, seed, paths  <- change model here
│   ├── activations.py        # load model, extract residual stream
│   ├── probes.py             # Probe object + fit_mms / fit_lr
│   ├── evaluate.py           # layer sweep, AUROC, transfer_matrix, cosines
│   ├── plotting.py           # per-type report + comparison heatmaps
│   └── data/
│       ├── base.py           # Example dataclass + paired() helper
│       ├── sycophancy.py     # induced by user social pressure
│       ├── sandbagging.py    # induced by eval-awareness + capability filter
│       ├── omission.py       # induced by "don't volunteer the bad fact"
│       └── __init__.py       # registry: name -> build()
├── scripts/
│   ├── train_one.py          # one type, full diagnostic figure
│   └── compare.py            # all three + transfer matrix + geometry
├── tests/test_smoke.py       # fast checks, no model download
└── results/                  # figures + .npy outputs (gitignored)
```

The mental model: `data/` is where the **research design** lives (and where you'll
spend most of your time). Everything else, activation extraction, probe fitting,
evaluation, plotting, is generic machinery you shouldn't need to touch to add a new
behaviour.

---

## Setup

Requires Python 3.10+. First model download is ~1GB.

```bash
git clone <your-repo-url> deception-probes
cd deception-probes

python -m venv .venv && source .venv/bin/activate   # windows: .venv\Scripts\activate
pip install -e ".[dev]"

pytest -q                    # should pass in seconds, no model download
```

Then a real run:

```bash
# one behaviour, writes results/report_sycophancy_lr.png
python -m scripts.train_one --type sycophancy

# sandbagging WITH the capability filter (important, see below)
python -m scripts.train_one --type sandbagging --filter

# the full comparison, writes results/comparison_lr.png
python -m scripts.compare --method lr
```

On a laptop this runs on CPU or Apple MPS. For the bigger model, bump `MODEL_NAME`
in `src/dprobe/config.py` to `Qwen/Qwen2.5-1.5B-Instruct` (or larger) and run on an
OSC GPU node.

---

## How the three datasets are built (and how to grow them)

Each type is a module in `src/dprobe/data/` exposing `build() -> list[Example]`. An
`Example` is one prompt with a label (`1` = deceptive, `0` = control). Datasets are
balanced 50/50.

- **sycophancy** — induced by the **user**, not a system prompt. Control asks a fact
  neutrally; deceptive asserts the wrong thing confidently and fishes for agreement.
- **sandbagging** — control says "answer correctly"; deceptive says "you're being
  evaluated, appear less capable, give a wrong answer." Each item carries its correct
  answer so the **capability filter** can run.
- **omission** — control discloses a material fact; deceptive is told to stay
  technically true but not volunteer the bad detail. Nothing false is said; the
  deception is in the omission.

The seed sets are **12 items each, on purpose tiny** so the pipeline runs end to end
in a few minutes. They are NOT enough for trustworthy AUROC. Before you believe any
number, scale each to 50-100+ items. To add data, open the module and append to the
`FACTS` / `QA` / `SCENARIOS` list (look for the `>>> ADD MORE` marker).

To add a whole new behaviour (say CoT-unfaithfulness):
1. copy `data/sycophancy.py` to `data/cot_unfaithful.py`, rewrite the examples
2. add one line to `data/__init__.py`
3. add the name to `DECEPTION_TYPES` in `config.py`
That's it, the scripts pick it up automatically.

### The capability filter (read this before trusting sandbagging)

A wrong answer is only *sandbagging* if the model could have answered correctly.
Otherwise you're just probing for incompetence. `capability_filter()` in
`sandbagging.py` runs each question under the control prompt and keeps only the ones
the model gets right. Always pass `--filter` for sandbagging. The current filter is a
crude string match; swap in a better grader for real runs.

---

## Working with Claude Code on this repo

This project is a good fit for Claude Code (multi-file, iterative). A few things that
help:

- `CLAUDE.md` in the root is loaded into Claude's context every session, so it already
  knows the layout and conventions. Update it when conventions change. Put private
  notes (scratch ideas, sandbox URLs) in `CLAUDE.local.md`, which is gitignored.
- Run `/init` once if you want Claude to regenerate/expand `CLAUDE.md` from the code.
- Good first asks: "scale the sycophancy dataset to 60 items pulling from
  TruthfulQA-style wrong-belief templates," or "add a pooled probe trained on all
  three types and add it as a 4th row/column in the transfer matrix."
- Keep changes small and run `pytest -q` after each one. The smoke test catches a
  broken refactor in seconds.

---

## Conventions (so two people don't fight the codebase)

- Label `1` is always the deceptive condition, `0` is always the control. Never flip.
- A matched pair differs **only** in what induces the behaviour. Same task underneath.
- New behaviour = new `data/` module + one registry line. Don't special-case a type
  anywhere else.
- Probes are layer-specific; a `Probe` object carries its own `.layer`.
- Outputs land in `results/` (gitignored). Commit figures you want to keep by adding
  them explicitly.

---

## Status / TODO

- [x] generic pipeline (activations, probes, layer sweep, AUROC, plots)
- [x] three seed datasets + capability filter
- [x] transfer matrix + direction geometry
- [ ] scale datasets to 50-100+ each
- [ ] add a pooled "general deception" probe as a baseline
- [ ] behavioural check for omission (does the generated answer actually omit?)
- [ ] move to Qwen2.5-1.5B (or larger) on OSC and rerun
