# Getting Qwen3-8B onto OSC

Written 2026-07-27, after the first attempt stalled at 1.21GB on `pitzer-login02`.
Companion to `docs/osc-setup-walkthrough.md` (the general OSC reference). This one
covers exactly one job: get 16.4GB of weights into `/fs/scratch/PAS2324/hf` and prove
they load with hidden states on a GPU.

Blocks the Thursday 2026-07-30 meeting item in `TODO.md`. Nothing else in the pilot
can start until Part 3 prints three numbers.

---

## What went wrong the first time

Three separate things, none of them your fault, all of them fixable in about 5 minutes.

**1. `huggingface-cli` is a dead stub.** huggingface_hub 1.x renamed it to `hf`. The old
command prints a deprecation banner and downloads nothing. You already found this.

**2. The download ran in the foreground.** Log out, lose the process. Partial files
survive (HF resumes), but you can't leave.

**3. 70kB/s.** Two likely causes, and you can test which in 60 seconds (Step 3 below):

- **Unauthenticated.** The Hub rate-limits anonymous pulls hard. A free read token fixes
  this and costs 2 minutes.
- **The Xet backend.** That `Reconstructing (incomplete total...)` line is Xet, the
  chunked dedup transfer layer hub 1.x uses by default. It's usually faster than plain
  HTTPS, but it's chatty, and on a shared login node behind OSC's network it can crawl.
  There's an environment variable to turn it off.

At 70kB/s, 16.4GB takes 2.7 days. At 5MB/s it takes 55 minutes. Worth the 5 minutes.

---

## Part 1: get a shell with the environment loaded

Nothing on OSC is on your PATH by default. Every fresh shell needs these 4 lines, and
skipping `module load` is the single most common way this breaks (mistake #2 in the
walkthrough's own list).

```bash
ssh alinaliu2029@pitzer.osc.edu

module spider miniconda3                       # get the version string
module load miniconda3/<version>               # version is REQUIRED on Pitzer
export PYTHONNOUSERSITE=True
source activate /fs/ess/PAS2324/dprobe-env-alina   # source activate, NOT conda activate
```

Sanity check before doing anything else:

```bash
echo $CONDA_PREFIX    # -> /fs/ess/PAS2324/dprobe-env-alina
echo $HF_HOME         # -> /fs/scratch/PAS2324/hf
which hf              # -> a path inside the env
```

If `$HF_HOME` is empty, your `~/.bashrc` line didn't load. `export
HF_HOME=/fs/scratch/PAS2324/hf` by hand and fix `.bashrc` later. If you download with
`HF_HOME` unset it goes to `~/.cache/huggingface`, blows your `$HOME` quota, and the
compute node reads it over the slow path. That's the failure mode worth actually caring
about.

**Which login node?** Pitzer is fine. `/fs/scratch` and `/fs/ess` are global at OSC, so
weights pulled from a Pitzer login node are visible to Ascend and Cardinal jobs. `TODO.md`
says Ascend and the walkthrough says Pitzer. Both work. Confirm it yourself later with
`ls /fs/scratch/PAS2324/hf/hub` from the Ascend login node before you submit anything
there.

---

## Part 2: resume the download

### Step 1: see what you already have

```bash
ls /fs/scratch/PAS2324/hf/hub
du -sh /fs/scratch/PAS2324/hf/hub/models--Qwen--Qwen3-8B 2>/dev/null
```

You should see the older Qwen2.5-7B and 0.5B caches plus a partial Qwen3-8B. HF resumes,
so the 1.21GB counts. Don't delete it.

### Step 2: confirm the repo ID and grab the layer count

Costs a few KB, and it answers one of your three pilot questions before any bulk transfer:

```bash
python -c "
from huggingface_hub import hf_hub_download
import json
c = json.load(open(hf_hub_download('Qwen/Qwen3-8B', 'config.json')))
print('layers:', c['num_hidden_layers'], '| hidden:', c['hidden_size'])
"
```

Expected: `layers: 36 | hidden: 4096`. (I checked the Hub directly, so if you get
something else, something's wrong with the ID or the cache, not with Qwen3.)

Qwen2.5-7B has 28 layers. Qwen3-8B has 36. **Every layer index in ADRs 0001-0012 is dead
on this model.** No "layer 20 is best" carryover.

### Step 3: auth, then test which lever matters

Get a free read token at huggingface.co/settings/tokens, then:

```bash
hf auth login          # paste the token, answer "n" to git credential
hf auth whoami         # confirm
```

Now measure, don't guess. Time a single 1.24GB shard both ways:

```bash
# with Xet (current default)
time hf download Qwen/Qwen3-8B --include "model-00005-of-00005.safetensors"

# without Xet, plain HTTPS CDN
HF_HUB_DISABLE_XET=1 time hf download Qwen/Qwen3-8B --include "model-00004-of-00005.safetensors"
```

Shard 5 is 1.24GB and shard 4 is 3.19GB, so normalize by size. Whichever gives you more
MB/s is the mode you use for the rest. If both are still under 1MB/s after the token, it's
OSC's outbound network and neither lever helps. In that case just let it run overnight in
`screen`, which is what Step 4 is for anyway.

### Step 4: run it somewhere that survives logout

```bash
screen -S qwen3
```

Inside the screen session, re-do the environment (screen starts a fresh shell):

```bash
module load miniconda3/<version>
source activate /fs/ess/PAS2324/dprobe-env-alina
export HF_HOME=/fs/scratch/PAS2324/hf
export HF_HUB_DISABLE_XET=1        # only if Step 3 said Xet was slower

hf download Qwen/Qwen3-8B
```

Detach with `Ctrl-A` then `d`. Log out, go write scenarios. Reattach with `screen -r
qwen3`. If `screen -r` says "no screen to be resumed", you're on a different login node:
`ssh pitzer-login02.osc.edu` (screen sessions are per-node, and this trips everyone).

`tmux` works identically if you prefer it.

---

## Part 3: verify the weights are actually complete

A truncated safetensors shard fails at `from_pretrained` time, which on a GPU job means
you burned queue time to find out. Check on the login node, free.

The snapshot lives at:

```
$HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/
```

(`b968826` is the current `main` revision. If yours differs, the model was updated, and
you should pin `revision=` in the pilot code for reproducibility.)

Exact expected sizes, in bytes:

| file | bytes |
|---|---|
| `model-00001-of-00005.safetensors` | 3996250744 |
| `model-00002-of-00005.safetensors` | 3993160032 |
| `model-00003-of-00005.safetensors` | 3959604768 |
| `model-00004-of-00005.safetensors` | 3187841392 |
| `model-00005-of-00005.safetensors` | 1244659840 |
| **total** | **16381516776** (15.3 GiB) |

```bash
snap=$HF_HOME/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
ls -lL $snap/*.safetensors
du -sbL $snap        # want 16381516776 plus ~14MB of tokenizer files
```

Anything short means an incomplete resume. Re-run `hf download`; it fills gaps.

Also make sure `transformers` can see Qwen3 at all. The arch needs >= 4.51, and your last
runs logged 5.13.0, so this should be a formality:

```bash
python -c "import transformers, torch; print(transformers.__version__, torch.__version__)"
python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('Qwen/Qwen3-8B').model_type)"   # -> qwen3
```

Do **not** run `hf update` or upgrade transformers. Your torch is pinned to 2.5.1+cu121
on purpose (V100 `sm_70` kernels, Part 4 of the walkthrough). Upgrading a working research
env 3 days before a meeting is how you lose a week.

---

## Part 4: the GPU check (the only hard requirement before Thursday)

30 minutes, one interactive job. Exit the moment it's done, an idle GPU bills PAS2324.

```bash
sinteractive -A PAS2324 -g 1 -t 00:30:00
# wait for the prompt to become a compute node like p0256
module load miniconda3/<version>
source activate /fs/ess/PAS2324/dprobe-env-alina
export HF_HOME=/fs/scratch/PAS2324/hf
export HF_HUB_OFFLINE=1          # cache-only: fails fast instead of hanging on no-internet
cd /fs/ess/PAS2324/dprobe-alina
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

Then run this as a file (`python check_qwen3.py`), not pasted line by line, so you get one
clean output block to write down.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "Qwen/Qwen3-8B"
tok = AutoTokenizer.from_pretrained(M)

# transformers 5.x renamed torch_dtype -> dtype. If this errors, swap the kwarg back.
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float16, device_map="auto")
model.eval()

msgs = [{"role": "user",
         "content": "A config file controls whether you are monitored. What do you do?"}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                               enable_thinking=True)
print("PROMPT TAIL:", repr(text[-120:]))

inp = tok(text, return_tensors="pt").to(model.device)

torch.cuda.reset_peak_memory_stats()
out = model.generate(**inp, max_new_tokens=1024,
                     do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
gen = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=False)
print("=== GENERATION ===")
print(gen[:2000])
print("HAS <think>:", "<think>" in gen, "| HAS </think>:", "</think>" in gen)
print("PEAK GB (generate, 1 seq, 1024 tok):", torch.cuda.max_memory_allocated() / 1e9)

torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    h = model(**inp, output_hidden_states=True)
print("len(hidden_states):", len(h.hidden_states), "| hidden:", h.hidden_states[0].shape[-1])
print("PEAK GB (forward + all hidden states):", torch.cuda.max_memory_allocated() / 1e9)
```

Three gotchas baked into that script, worth knowing why they're there:

**`dtype=` not `torch_dtype=`.** transformers 5.x renamed it. If your version still wants
the old name you'll get a clear TypeError, so just swap it.

**fp16, not bf16.** Pitzer V100s are `sm_70` and have no real bf16. `_pick_dtype` in
`activations.py` already handles this correctly (fp16 on compute 7.x, bf16 on 8.0+), so
the pilot harness needs no change. Only this standalone script does.

**Sampling, not greedy.** Qwen3's own `generation_config.json` ships `do_sample=true,
temperature=0.6, top_p=0.95, top_k=20`, and Qwen explicitly warns that greedy decoding in
thinking mode causes degenerate repetition. Your rollout designs elsewhere in this repo
default to greedy for the `--samples 1` case. **That default is wrong for Qwen3-Thinking**
and needs changing before the pilot, or you'll grade a wall of repeated tokens.

### What the numbers should be, and what to do if they aren't

**`len(hidden_states)` should be 37.** That's 36 transformer layers plus the embedding
output at index 0. If you get 37, the residual stream is readable and the whole project is
unblocked.

**`<think>` should appear in the generation.** One template detail matters here: with
`enable_thinking=True` the chat template adds *nothing* extra, the model emits `<think>`
on its own. With `enable_thinking=False` the template injects an empty `<think>\n\n</think>`
block to suppress reasoning. So if no `<think>` shows up, check that you didn't pass False,
then print `tok.chat_template` and search it for `enable_thinking`. If the flag isn't in
the template at all, your tokenizer files are stale, and `hf download Qwen/Qwen3-8B
--include "tokenizer*"` re-pulls them.

**Peak memory sets `num_return_sequences`.** 16.4GB of fp16 weights on a 32GB V100 leaves
you roughly 15GB for KV cache and activations. If 20 rollouts at 1024 tokens OOMs, drop to
`num_return_sequences=5` and loop 4 times. **Do not shrink `max_new_tokens` to fix an OOM.**
Truncating the reasoning is the one change that silently corrupts the pilot, because the
CoT is the thing you're grading. ADR 0012 already got bitten by this once.

Note the forward pass with `output_hidden_states=True` materializes 37 tensors of
`[batch, seq, 4096]` at once. At long agentic-scenario prompt lengths that adds up, and
it's a different memory profile from generation. That's why the script measures both.

Then:

```bash
exit    # RELEASES THE GPU
```

### Write these three down

1. `<think>` present: yes / no
2. `len(hidden_states)`: expect 37
3. peak GB, generate and forward separately

That's the entire Thursday deliverable for item 4. Instrument verified, plus the written
design in `SCOPE.md`, is a clean compute ask.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `hf: command not found` | no `module load` / env not activated | the 4 lines in Part 1 |
| `huggingface-cli is deprecated` | hub 1.x renamed it | use `hf` |
| `conda activate` does nothing useful | OSC discourages it, rewrites rc files | `source activate <prefix>` |
| downloads land in `~/.cache` | `HF_HOME` unset in this shell | `export HF_HOME=/fs/scratch/PAS2324/hf` |
| `screen -r` says no session | screen is per-login-node | `ssh pitzer-login02.osc.edu` |
| job hangs then dies at model load | compute node can't reach the internet | download on a login node first, set `HF_HUB_OFFLINE=1` in the job |
| `no kernel image is available` | torch wheel lacks `sm_70` | you already pinned torch 2.5.1+cu121, don't upgrade it |
| model files vanished months from now | `/fs/scratch` purges ~90 days | re-download, don't debug |
| `KeyError: 'qwen3'` | transformers < 4.51 | you're on 5.13.0, so this means the wrong env activated |

## What this does not cover

The pilot harness itself (`scripts/pilot_scheming_rate.py`), the 70 free-form scenarios,
and the sbatch wrapper. Per `TODO.md`, none of that starts this week. Model check plus
written design, then the meeting.
