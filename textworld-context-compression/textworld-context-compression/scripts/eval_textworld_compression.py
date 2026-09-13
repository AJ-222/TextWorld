"""Multi-episode evaluation on the held-out crafting distribution (Sec 3.4.2), with
the Episode-3 IQM + bootstrap-CI protocol from Sec 4.1 (Weeks 20-21) / H1.

Runs `n_seeds` independent 3-episode rollouts (Sec 3.3: 3 episodes of in-context
adaptation) of a given agent+checkpoint against the craft-test game pool, and reports:
  - per-episode success rate (episode 1/2/3)
  - Episode-3 Interquartile Mean (IQM) across seeds, with a percentile-bootstrap 95%
    CI (Sec 4.2, success criterion #1: "non-overlapping 95% confidence intervals
    across 5 evaluation seeds")
  - malformed-<summarise/> rate and forced-fallback-compression rate (useful
    diagnostics for the RQ1 "does the model actually learn to use the compression
    action well" question, and for reading the No-Shielding-style ablations even
    though this project dropped the KV-quantisation layer)

Deliberately uses plain HF `generate()` rather than standing up a vLLM server —
simpler to run ad hoc against a training checkpoint, at the cost of being slower
than the vLLM-served rollouts used during GRPO training itself. Fine for the
evaluation-loop cadence in Sec 4.1 (Weeks 14-17), not meant for the training loop.

If the `rliable` package is installed, we use its `metrics.aggregate_iqm` /
bootstrap utilities directly (closer to the literature-standard implementation cited
in Sec 4.1); otherwise we fall back to a small local implementation
(`_iqm_with_ci`) that computes the same statistic, so this script runs either way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ORBIT_ROOT = REPO_ROOT.parent
if str(ORBIT_ROOT) not in sys.path:
    sys.path.insert(0, str(ORBIT_ROOT))

from agents.compressive_text_agent import CompressiveTextAgent  # noqa: E402
from agents.heuristic_summary_agent import HeuristicSummaryAgent  # noqa: E402
from envs.textworld_env_adapter import TextWorldEnvAdapter  # noqa: E402

AGENT_CLASSES = {
    "compressive_agent": CompressiveTextAgent,
    "heuristic_summary_agent": HeuristicSummaryAgent,
    "static_append_agent": lambda **kw: HeuristicSummaryAgent(threshold=None, **kw),
}


def _iqm_with_ci(scores: List[float], n_bootstrap: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Interquartile mean + percentile bootstrap 95% CI, no external deps.

    IQM = mean of the middle 50% of `scores` (i.e. drop the bottom and top quartile,
    average what's left) — more robust to outlier seeds than a plain mean, and the
    statistic the proposal asks for (Sec 4.1/4.2, citing Agarwal et al. 2021).
    """
    import random

    def iqm(xs: List[float]) -> float:
        if not xs:
            return 0.0
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        lo = n // 4
        hi = n - lo
        middle = xs_sorted[lo:hi] if hi > lo else xs_sorted
        return sum(middle) / len(middle)

    point = iqm(scores)
    rng = random.Random(seed)
    boot = []
    n = len(scores)
    for _ in range(n_bootstrap):
        sample = [scores[rng.randrange(n)] for _ in range(n)]
        boot.append(iqm(sample))
    boot.sort()
    lo_idx = int(0.025 * len(boot))
    hi_idx = int(0.975 * len(boot))
    return {"iqm": point, "ci_low": boot[lo_idx], "ci_high": boot[min(hi_idx, len(boot) - 1)]}


def run_rollout(
    model,
    tokenizer,
    agent_class_name: str,
    games_dir: str,
    split: str,
    seed: int,
    max_turns_per_episode: int,
    n_episodes: int,
    max_new_tokens: int = 512,
) -> Dict:
    import torch

    agent_ctor = AGENT_CLASSES[agent_class_name]
    agent = agent_ctor(max_steps=max_turns_per_episode * n_episodes + 20)

    env = TextWorldEnvAdapter(
        env_kwargs={
            "games_dir": games_dir,
            "split": split,
            "seed": seed,
            "max_turns": max_turns_per_episode * n_episodes,
            "curriculum_enabled": False,  # Sec 3.6: no shaping at eval time
        }
    )
    obs, info = env.reset(seed=seed)

    episode_successes: List[bool] = []
    malformed_count = 0
    forced_fallback_count = 0

    reward, done, step_info = 0.0, False, {}
    turns = 0
    max_total_turns = max_turns_per_episode * n_episodes + 10
    while not done and turns < max_total_turns:
        turns += 1
        agent.update_from_env(obs, reward, done, step_info)

        prompt_ids = tokenizer.apply_chat_template(
            agent.chat_completions, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        with torch.no_grad():
            out_ids = model.generate(
                prompt_ids.to(model.device),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out_ids[0][prompt_ids.shape[1]:], skip_special_tokens=True)

        action = agent.update_from_model(response)
        last_step_info = agent.get_current_state().info
        if last_step_info.get("malformed_summary_tag"):
            malformed_count += 1
        if getattr(agent, "_last_compression_reason", None) == "forced_threshold":
            forced_fallback_count += 1

        obs, reward, done, step_info = env.step(action)
        if step_info.get("episode_done") or step_info.get("terminated"):
            if "episode_success" in step_info:
                episode_successes.append(bool(step_info["episode_success"]))

    env.close()
    return {
        "seed": seed,
        "episode_successes": episode_successes,
        "malformed_summary_turns": malformed_count,
        "forced_fallback_compressions": forced_fallback_count,
        "total_turns": turns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--agent", choices=list(AGENT_CLASSES), default="compressive_agent")
    parser.add_argument("--games-dir", default=str(REPO_ROOT / "games"))
    parser.add_argument("--split", default="test_craft")
    parser.add_argument("--n-seeds", type=int, default=5)  # Sec 4.2: "5 evaluation seeds"
    parser.add_argument("--seed-start", type=int, default=9000)
    parser.add_argument("--max-turns-per-episode", type=int, default=40)
    parser.add_argument("--n-episodes", type=int, default=3)  # Sec 3.3
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    per_seed_results = []
    for i in range(args.n_seeds):
        seed = args.seed_start + i
        result = run_rollout(
            model,
            tokenizer,
            args.agent,
            args.games_dir,
            args.split,
            seed,
            args.max_turns_per_episode,
            args.n_episodes,
        )
        per_seed_results.append(result)
        print(f"[seed {seed}] episode_successes={result['episode_successes']}")

    episode3_scores = [
        float(r["episode_successes"][2]) if len(r["episode_successes"]) > 2 else 0.0
        for r in per_seed_results
    ]
    stats = _iqm_with_ci(episode3_scores)

    summary = {
        "agent": args.agent,
        "model_path": args.model_path,
        "split": args.split,
        "n_seeds": args.n_seeds,
        "episode3_iqm": stats["iqm"],
        "episode3_ci95": [stats["ci_low"], stats["ci_high"]],
        "per_seed": per_seed_results,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "per_seed"}, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote full results to {args.out}")


if __name__ == "__main__":
    main()
