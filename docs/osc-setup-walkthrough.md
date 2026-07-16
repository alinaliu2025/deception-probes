# OSC setup walkthrough — what I did, every command, and why

Personal reference written 2026-07-09, covering the full session where I synced
Jack's branch and got running on the Ohio Supercomputer Center (OSC). Companion
to `docs/HPC.md` (Jack's cheatsheet) — this one explains each step from zero.

**What was accomplished:** local repo synced to `sycophancy_v2`; OSC account
verified on project PAS2324; GitHub SSH access from OSC; repo cloned to project
storage; personal conda env built; V100 PyTorch incompatibility diagnosed and
fixed; pipeline smoke-tested end to end on a GPU; run record pushed to GitHub;
batch script added so long runs don't need babysitting.

---

## Part 1 — Getting Jack's work locally (on my Mac)

Jack's work was on git *branches*, not `main`.

```bash
cd ~/Desktop/2026Research/deception-probes
git fetch origin          # downloads all remote branches/commits; changes no files
git switch sycophancy_v2  # points my working tree at his newest branch
pip install -e ".[dev]"   # branch added pyproject.toml + new modules → reinstall
python -m pytest -q       # fast self-checks, no model download
```

Why `sycophancy_v2`: it contains every other branch he pushed
(`first-sycophancy`, `sycophancy-no-leak`, `sycophancy-factual`) plus the
steering work. My `main` is untouched; merging into `main` waits for team
agreement (`git switch main && git merge sycophancy_v2`).

If `git switch` complains about untracked files that would be overwritten:
those were leftover duplicates (from a permission-blocked checkout attempt);
removing them and retrying was safe because they were byte-identical to the
branch's tracked copies.

Stale lock files (`.git/index.lock`, `.git/packed-refs.lock`, etc.) block every
git command with "another git process seems to be running." If no git process
is actually running, deleting the lock files is the fix:

```bash
rm -f .git/index.lock .git/packed-refs.lock .git/objects/pack/tmp_*
```

---

## Part 2 — OSC concepts worth committing to memory

**Clusters.** OSC has three GPU clusters; the team uses them as tiers:

| Cluster | GPU | Use |
|---|---|---|
| Pitzer | V100 (older) | smoke tests, debugging |
| Ascend | A100 | formal experiment runs |
| Cardinal | H100 | opportunistic, if queue is short |

**Login nodes vs compute nodes.** SSHing into `pitzer.osc.edu` or
`ascend.osc.edu` lands on a *login node*: a shared machine for editing, git,
and submitting jobs. It has no GPU, computing on it is against the rules, and
it is **free**. Real work happens on *compute nodes* (hostnames like `p0256`),
which you only reach through the scheduler.

**SLURM (the scheduler).** All compute is requested through SLURM, two ways:

- `sinteractive -A PAS2324 -g 1` — an interactive shell on a GPU node. Good for
  debugging. Bills the project the entire time it's held, even idle, and dies
  when the connection drops or walltime expires.
- `sbatch script.sbatch` — a batch job. Queues, runs unattended, survives
  logoff, bills only while running. **The right choice for anything over ~30 min.**

**Billing (lab funds).** Only SLURM jobs cost money, charged to `PAS2324`.
Login-node time and storage-within-quota are free. Rules of thumb: never hold
an idle interactive GPU session (exit as soon as done); give batch jobs a
realistic `--time` cap so a hung job can't burn 8 hours.

**Storage — three areas, different rules:**

| Path | Properties | Use for |
|---|---|---|
| `$HOME` | tiny quota, slow from compute nodes | dotfiles only |
| `/fs/scratch/PAS2324` | big, fast, **purged ~90 days** | model weight cache (`HF_HOME`) |
| `/fs/ess/PAS2324` | persistent project storage | code clones, envs, results |

**The module system.** Software isn't on PATH by default; `module load <name>`
activates it. Pitzer requires an explicit version
(`module load miniconda3/<version>`); `module spider miniconda3` lists what's
available. Ascend's versions can differ — re-check per cluster.

---

## Part 3 — One-time OSC setup, command by command

### 3.1 Log in and verify project access

```bash
ssh alinaliu2029@pitzer.osc.edu   # OSC credentials + Duo push
OSCfinger $USER                   # must list PAS2324 under SLURM accounts
```

### 3.2 GitHub SSH key

GitHub no longer accepts passwords over the command line; the OSC machine needs
its own key pair.

```bash
ssh-keygen -t ed25519             # Enter through prompts; makes a key PAIR
cat ~/.ssh/id_ed25519.pub         # the PUBLIC key: one line, "ssh-ed25519 AAAA…"
```

Paste that line into github.com → Settings → SSH and GPG keys → New SSH key.
Verify: `ssh -T git@github.com` → "Hi alinaliu2025!".

**Lesson learned the confusing way:** `id_ed25519` (no extension) is the
*private* key — begins `-----BEGIN OPENSSH PRIVATE KEY-----`, never leaves the
machine, never gets pasted anywhere, never gets sent to anyone. Only the `.pub`
file is shared. If the `.pub` is ever missing, regenerate it:
`ssh-keygen -y -f ~/.ssh/id_ed25519 > ~/.ssh/id_ed25519.pub`.

### 3.3 Clone the repo to project storage

```bash
cd /fs/ess/PAS2324
git clone git@github.com:alinaliu2025/deception-probes.git dprobe-alina
cd dprobe-alina
git switch sycophancy_v2
```

`/fs/ess` because it persists and has room; `dprobe-alina` to keep my clone
separate from Jack's.

### 3.4 Build a personal conda environment

```bash
module load miniconda3/<version>              # version from `module spider miniconda3`
conda config --remove channels defaults       # avoid Anaconda default-channel issues
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create --prefix /fs/ess/PAS2324/dprobe-env-alina python=3.10
export PYTHONNOUSERSITE=True                  # block stray ~/.local packages
source activate /fs/ess/PAS2324/dprobe-env-alina
pip install -e ".[dev]"                       # run from inside dprobe-alina
```

Key details: use `source activate` — **not** `conda activate`, which rewrites
shell config files (OSC-recommended practice). `--prefix` puts the env on
persistent project storage instead of `$HOME`. My own env (rather than Jack's
shared one) because his env's editable install points at *his* clone.

### 3.5 Keep model downloads off $HOME

```bash
echo 'export HF_HOME=/fs/scratch/PAS2324/hf' >> ~/.bashrc
source ~/.bashrc
```

Hugging Face caches model weights (15–30 GB per large model) wherever `HF_HOME`
points. Scratch is fast and big; a purge just means re-downloading.

---

## Part 4 — The V100/PyTorch incident (worth remembering)

**Symptom:** first GPU run crashed with
`CUDA error: no kernel image is available for execution on the device`.

**Cause:** Pitzer's V100s are compute capability `sm_70`. New PyTorch wheels
from a plain `pip install torch` no longer ship `sm_70` kernels — the code has
literally no machine code for that GPU. Diagnosis:

```bash
nvidia-smi | head -5                                            # which GPU am I on?
python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
```

If `sm_70` is absent from the arch list on a V100, this exact error follows.

**Fix:** install an older wheel that still includes V100 support:

```bash
pip uninstall -y torch
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

torch 2.5.1/cu121 runs on V100, A100, and H100 alike — one env works on every
OSC tier. Anyone building a fresh env from the repo (torch is unpinned in
requirements) will hit this; it should get pinned or documented in `docs/HPC.md`.

---

## Part 5 — Running the pipeline on a GPU

### 5.1 Interactive smoke test (what I did)

```bash
sinteractive -A PAS2324 -g 1      # request 1 GPU, charged to the lab project
# …wait for allocation; prompt changes to a compute node like p0256…
module load miniconda3/<version>  # compute nodes start with a clean shell
source activate /fs/ess/PAS2324/dprobe-env-alina
cd /fs/ess/PAS2324/dprobe-alina
python -m pytest -q
python -m scripts.train_one --type sycophancy --design rollout --source factual-small --filter
exit                              # RELEASES THE GPU — never leave it idle
```

`--source factual-small` uses a tiny offline fixture bundled in the repo: no
downloads, minutes to run, proves the plumbing only. **Its AUROC means nothing.**

### 5.2 Where results go

Each run writes `results/runs/<UTC-timestamp>_<kind>_<method>_<git-sha>_<node>/`:

- `meta.json` — config, flags, model, AUROC, filter stats. The **only
  git-tracked** artifact; committing it is how runs become visible to the team.
- `run_log.txt` — per-question rollout log (gate result, caved/held, kept sets).
- `.png` plots and `probe.npz` — gitignored, heavy, regenerable; `scp` them back
  if needed:

```bash
# from the Mac:
scp -r alinaliu2029@pitzer.osc.edu:/fs/ess/PAS2324/dprobe-alina/results/runs/<run-folder> \
    ~/Desktop/2026Research/deception-probes/results/runs/
```

### 5.3 Publish the run record

```bash
run=$(ls -t results/runs/ | head -1)         # newest run folder
git add results/runs/$run/meta.json
git commit -m "OSC smoke run: rollout factual-small, V100"
git push origin sycophancy_v2
```

My verified smoke run: `2026-07-09T19-31-30Z_sycophancy_lr_21d07f8_p0256`
(the SHA `21d07f8` in the name = the commit the code ran at; `p0256` = the node).

---

## Part 6 — Batch jobs (the right way to run anything long)

The real experiment (7B model, `--source factual`, hours of generation) must be
a batch job: interactive sessions die on disconnect or walltime and bill while
idle. The script lives at `scripts/osc/train_one.sbatch`.

Anatomy of the `#SBATCH` header:

| Line | Meaning |
|---|---|
| `--account=PAS2324` | who pays (mandatory on OSC) |
| `--time=08:00:00` | walltime cap; job is killed after this — also caps the spend |
| `--gpus-per-node=1` | request one GPU (OSC prefers this over `--gres`) |
| `--output=results/slurm-%j.out` | live stdout/stderr, `%j` = job ID |
| `--mail-type=END,FAIL` + `--mail-user` | email me when it finishes or dies |

The script body redoes what a fresh shell needs (module load, `source
activate`, `HF_HOME`, `cd`) because batch jobs start clean, then runs
`train_one`. Extra flags pass through: `sbatch scripts/osc/train_one.sbatch
--rollouts 16`. Model override:
`sbatch --export=ALL,DPROBE_MODEL=Qwen/Qwen2.5-14B-Instruct scripts/osc/train_one.sbatch`.

Submit → monitor → collect:

```bash
ssh alinaliu2029@ascend.osc.edu       # Ascend = A100 = formal tier
cd /fs/ess/PAS2324/dprobe-alina && git pull
sbatch scripts/osc/train_one.sbatch   # returns a job ID; safe to log off now
squeue -u $USER                       # queue state (PD = pending, R = running)
tail -f results/slurm-<jobid>.out     # watch live output
scancel <jobid>                       # kill a job
```

When the email arrives: log in, check the new `results/runs/` folder, commit
its `meta.json`, push.

---

## Every-new-session checklist ("rebooting")

Fresh shells (login or compute) remember nothing except `~/.bashrc`. Each time:

```bash
ssh alinaliu2029@pitzer.osc.edu        # or ascend.osc.edu
module load miniconda3/<version>
source activate /fs/ess/PAS2324/dprobe-env-alina
cd /fs/ess/PAS2324/dprobe-alina
git pull                               # sync latest before running anything
```

(`HF_HOME` is already exported by `~/.bashrc`.)

---

## Quick command reference

| Command | What it does |
|---|---|
| `OSCfinger $USER` | show my SLURM accounts (confirm PAS2324) |
| `module spider miniconda3` | list available versions of a module |
| `sinteractive -A PAS2324 -g 1` | interactive GPU shell (bills until exit) |
| `sbatch <file>.sbatch` | submit batch job |
| `squeue -u $USER` | my queued/running jobs |
| `scancel <jobid>` | cancel a job |
| `nvidia-smi` | which GPU I'm on + memory in use |
| `ls -t results/runs/ \| head -1` | newest run folder |

## Mistakes made this session, so they aren't repeated

1. **Almost shared a private key.** Only `.pub` files ever leave the machine.
2. **`module load miniconda3` without a version** fails on Pitzer — always
   check `module spider` first, and re-check on each cluster.
3. **Fresh `pip install torch` breaks on V100s** — pin `torch==2.5.1` (cu121)
   or run only on A100/H100.
4. **Started an hours-long run in an interactive session.** It would have died
   at walltime or disconnect. Long runs go through `sbatch`, always.
5. **Idle interactive sessions bill the project.** Exit the moment the work is
   done; login nodes are the free place to sit and think.
