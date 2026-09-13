# TextWorld Context-Compression Meta-RL — overlay on ORBIT

This is the narrowed-scope implementation from your Honours proposal (KIVI/KV-cache
quantisation layer cut — see the memory note from our planning chat): a GRPO-trained
agent that manages its own context via periodic `<summary>` blocks, evaluated
zero-shot on TextWorld under ORBIT's multi-episode meta-RL protocol.

It is **not a standalone repo** — it's an overlay meant to be dropped into a checkout
of [ORBIT](https://github.com/XiaofengLin7/ORBIT) (Lin et al. 2026), whose own
`third_party/rllm` (+ verl) training stack this project reuses unmodified. Copy this
folder's contents into your ORBIT checkout so the paths line up:

```
your-orbit-checkout/
├── agents/                    <- ORBIT's own agents, PLUS these new files:
│   ├── compressive_text_agent.py
│   ├── heuristic_summary_agent.py
│   └── _extractive_summary.py
├── envs/                      <- ORBIT's own envs, PLUS:
│   ├── textworld_env_adapter.py
│   ├── curriculum_reward.py
│   ├── _admissible_matching.py
│   └── _tw_threadsafety.py
├── prompts/
│   └── compression_prompts.py     <- new
├── data/
│   ├── generate_textworld_games.py    <- new
│   └── prepare_sft_data.py            <- new
├── scripts/
│   ├── train_textworld_compression.py <- new
│   ├── train_textworld_compression.sh <- new
│   ├── sft_warmstart.py               <- new
│   └── eval_textworld_compression.py  <- new
├── configs/
│   └── textworld_compression_config.yaml  <- new
└── tests/
    ├── test_compressive_text_agent.py     <- new
    └── test_textworld_env_adapter.py      <- new
```

```bash
cp -r envs agents prompts data scripts configs tests /path/to/your/ORBIT-checkout/
cd /path/to/your/ORBIT-checkout
```

## What's already verified vs. what's not

I don't have a GPU or network access to PyPI/TextWorld in the environment this was
written in, so here's the honest split:

**Verified by direct execution** (see `tests/test_compressive_text_agent.py` and the
pure-unit half of `tests/test_textworld_env_adapter.py` — both ran clean here):
- The core compression mechanic: `CompressiveTextAgent` correctly splices its own
  message history down to `[system, compressed_block]` on a policy-chosen
  `<summarise/>` turn, correctly appends the next observation after that, falls back
  to a deterministic extractive summary on a malformed `<summarise/>` (no `<summary>`
  tag), and force-compresses once accumulated tokens cross the threshold — all
  without crashing.
- `envs/_admissible_matching.py`'s boxed-action extraction, sentinel detection, and
  fuzzy admissible-command matching.
- `envs/curriculum_reward.py`'s graph-distance potential and shaping arithmetic
  (including the terminal-state Φ=0 special case).
- `HeuristicSummaryAgent` in both modes (`threshold=None` = static-append baseline,
  `threshold=N` = scripted heuristic-summarisation baseline).

**Written but NOT executed** (no TextWorld/rLLM/verl/GPU available here):
- `envs/textworld_env_adapter.py`'s actual TextWorld gym integration (the logic
  mirrors ORBIT's own `ALFWorldEnvAdapter` closely, but hasn't run against a live
  game).
- `data/generate_textworld_games.py`'s `GameMaker`-based navigation-game construction
  and the `tw-make tw-cooking` CLI invocation for the crafting test set — these are
  written from general TextWorld API knowledge, not a live install. **Run
  `python data/generate_textworld_games.py --smoke-test` first** (builds 2+2 tiny
  games and validates one is solvable) before generating the full 200+30 pool. If it
  fails, the traceback tells you exactly which TextWorld call needs adjusting —
  cross-check against `tw-make tw-cooking --help` and
  `python -c "from textworld.generator.maker import GameMaker; help(GameMaker)"` on
  your machine.
- `scripts/sft_warmstart.py`'s chat-template tokenization — run with
  `--max-examples 8 --dump-first-example` first and eyeball the decoded output
  before committing to the full dataset.
- The LoRA/GRPO hyperparameter flags in `train_textworld_compression.sh`
  (`lora_rank`, `lora_alpha`, `target_modules`) — written against verl's documented
  PEFT support, not checked against your installed verl version's actual config
  schema. Dry-run with `python scripts/train_textworld_compression.py --cfg job ...`
  first.

None of this is a criticism of the design, it's just where a second pair of eyes
(yours, or your supervisors') is most valuable before a long run — the payoff of
having code you can actually read and adjust yourself.

## Design decisions worth a second opinion

These are judgment calls I had to make where the proposal's prose was ambiguous —
flagged prominently in the code, repeated here so they don't get missed:

1. **`<summarise/>` is a mutually-exclusive per-turn choice** (compress OR act, not
   both in one turn), consuming one turn of the turn budget but not changing game
   state. See `envs/textworld_env_adapter.py` docstring point 1 for the alternative
   reading and why I picked this one (matches Fig 3.1 and the Sec 3.3 action-space
   union literally).
2. **The heuristic-summarisation baseline uses a scripted extractive summary**, not
   a second LLM call — this keeps it a genuinely "non-learned" comparison point and
   avoids needing to route a second model call through the training loop. See
   `agents/heuristic_summary_agent.py` docstring.
3. **Forced-fallback compression (>1024 tokens) is a hard deterministic override**,
   not a soft prompt nudge — if the policy hasn't summarised by then, the code
   compresses for it using the same extractive summariser as the heuristic baseline.
   This matches "deterministic fallback similar to the thresholding in SUPO" (Sec
   3.5.3) literally.
4. Added a `max_consecutive_summarise` safety valve (default 3) so a degenerate
   policy can't stall a trajectory by spamming `<summarise/>` forever. Not in the
   proposal — remove or tune if you don't want it.

## Suggested run order

```bash
# 0. Sanity-check TextWorld generation against your installed version
python data/generate_textworld_games.py --smoke-test

# 1. Generate the full game pools (Sec 3.4: 200 nav train, 30 craft test)
python data/generate_textworld_games.py --n-train 200 --n-test 30

# 2. Build the SFT warm-start dataset (Sec 3.5.2: 500 oracle trajectories)
python data/prepare_sft_data.py --n-trajectories 500

# 3. SFT warm-start (Sec 3.5.2/3.5.1: one epoch, LoRA rank 16)
python scripts/sft_warmstart.py --max-examples 8 --dump-first-example   # sanity check first
python scripts/sft_warmstart.py                                         # full run

# 4. GRPO training on top of the SFT checkpoint
MODEL_PATH=./checkpoints/sft_warmstart bash scripts/train_textworld_compression.sh

# 5. Evaluate (Episode-3 IQM + bootstrap CI across 5 seeds, Sec 4.1/4.2)
python scripts/eval_textworld_compression.py \
    --model-path ./checkpoints/sft_warmstart \
    --agent compressive_agent --split test_craft

# Baselines for comparison (Sec 3.8): static-append and heuristic-summarisation
# don't need GRPO training at all — eval the base/SFT checkpoint directly with
# --agent static_append_agent / --agent heuristic_summary_agent.
```

## What's simplified vs. the full proposal (by design — you cut this)

- No KIVI 2-bit KV-cache quantisation, no mixed-precision sink shielding, no RQ2/RQ3,
  no H2/H3. `TextWorldEnvAdapter` and `CompressiveTextAgent` only implement the
  semantic-compression layer (RQ1/H1).
- Baselines/ablations kept: Static-Append, Heuristic-Summarisation, No-Curriculum
  (`curriculum_enabled: false` in the config), No-RL/SFT-only (just don't run GRPO).
  Dropped (no quantisation layer to ablate): Unquantised-GRPO-Control, No-Shielding.
- `scripts/eval_textworld_compression.py` uses plain HF `generate()`, not vLLM —
  simpler for ad hoc eval, slower than production serving. Fine for the Sec 4.1
  evaluation-loop cadence; swap for a vLLM client if 5 seeds × 30 games × 3 episodes
  gets too slow.

## Next things to build (not yet started here)

- Weeks 20-21 statistical-analysis pass: the eval script gives you per-seed
  Episode-3 success + IQM/CI; you still need the stratified-bootstrap
  probability-of-improvement comparison across agents (baseline vs. compressive)
  that Sec 4.1 calls for. `rliable` (pip install rliable) has this built in if the
  small local `_iqm_with_ci` isn't enough.
- No-Curriculum and No-RL(SFT-only) ablation *runner scripts* (currently just "pass
  different config/checkpoint" — fine, but a small `run_ablations.sh` that runs all
  three plus the two baselines back-to-back and dumps one comparison table would
  save you re-typing the eval command five times).
