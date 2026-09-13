"""Build the SFT warm-start dataset (proposal Sec 3.5.2): 500 oracle TextWorld
trajectories from the navigation training distribution, with `<summarise/>` actions
interleaved automatically whenever accumulated history exceeds 512 tokens.

Why this matters: GRPO from the base checkpoint risks *silent advantage collapse* —
early rollouts rarely succeed, so if a whole sampled group returns 0 reward the
advantage normaliser (Eq. 3.3) divides ~0/~0 and the gradient signal vanishes. The SFT
warm-start exists purely to give the policy a working prior over (a) the TextWorld
admissible-action format and (b) the `<summary>` block format, so GRPO isn't starting
from a policy that never emits either correctly.

Implementation note — this DOGFOODS the real runtime classes rather than
reimplementing the compression bookkeeping a second time:

    for each oracle action in the walkthrough:
        agent.update_from_env(obs, ...)              # same class used at train/eval time
        if len(agent.chat_completions) tokens > 512 and we haven't *just* compressed:
            synthetic_response = "<summary>...</summary>\\n\\boxed{<summarise/>}"
        else:
            synthetic_response = "\\boxed{<oracle action>}"
        action = agent.update_from_model(synthetic_response)   # does the real compression
        env.step(action)                                        # advances the real game

The summary text itself is the same non-learned extractive summariser used by
`HeuristicSummaryAgent` and `CompressiveTextAgent`'s forced fallback (see
`agents/_extractive_summary.py`) — NOT a real LLM call (this script runs offline,
no model server). This is a reasonable bootstrap (the model only needs to imitate
the *format*, not learn good summarisation content yet — that's what GRPO is for),
but it does mean the SFT summaries are mechanically bland. If you have API access to
a stronger model and want richer SFT summaries, swap `extractive_summary(...)` below
for a real call — the rest of the pipeline doesn't care how the text was produced.

Output: JSONL at `data/sft_textworld.jsonl`, one line per assistant turn:
    {"messages": [...prefix chat turns...], "target": "<assistant text>",
     "gamefile": ..., "is_compression_turn": bool}
Consumed by `scripts/sft_warmstart.py` for a standard token-level CE loss over
`target`, conditioned on `messages`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents._extractive_summary import extractive_summary  # noqa: E402
from agents.compressive_text_agent import CompressiveTextAgent  # noqa: E402
from envs.textworld_env_adapter import TextWorldEnvAdapter  # noqa: E402
from prompts.compression_prompts import SFT_SUMMARY_TRIGGER_THRESHOLD, SUMMARY_TOKEN_CAP  # noqa: E402


def _get_walkthrough(env: TextWorldEnvAdapter) -> list[str]:
    """Pull the oracle command list off the underlying TextWorld gym env.

    Requires `extras=["walkthrough"]` in the adapter's EnvInfos — the shipped
    `TextWorldEnvAdapter._init_tw_env` doesn't request this by default (it's not
    needed at train/eval time and walkthroughs are meaningless once the curriculum
    shaping / no-shaping distinction matters), so this function reaches into the
    adapter's private `_tw_env` and re-requests it directly. If your installed
    TextWorld exposes walkthroughs differently, this is the one place to fix it —
    see the smoke test in `generate_textworld_games.py` for the same lookup.
    """
    import textworld

    infos = env._tw_env.unwrapped._infos if hasattr(env._tw_env, "unwrapped") else {}
    walkthrough = infos.get("extra.walkthrough") or infos.get("walkthrough")
    if walkthrough:
        return list(walkthrough[0]) if walkthrough and isinstance(walkthrough[0], list) else list(walkthrough)

    # Fallback: re-register the same gamefile with an explicit walkthrough request.
    request_infos = textworld.EnvInfos(extras=["walkthrough"])
    env_id = textworld.gym.register_games([env._gamefile], request_infos, batch_size=1)
    probe = textworld.gym.make(env_id)
    _, infos = probe.reset()
    probe.close()
    walkthrough = infos.get("extra.walkthrough") or infos.get("walkthrough")
    if not walkthrough:
        raise RuntimeError(
            f"Could not retrieve a walkthrough for {env._gamefile}. Check that your "
            "TextWorld version supports EnvInfos(extras=['walkthrough'])."
        )
    return list(walkthrough[0]) if isinstance(walkthrough[0], list) else list(walkthrough)


def build_one_trajectory(games_dir: str, seed: int) -> list[dict]:
    env = TextWorldEnvAdapter(
        env_kwargs={"games_dir": games_dir, "split": "train_nav", "seed": seed, "max_turns": 200}
    )
    obs, info = env.reset(seed=seed)
    walkthrough = _get_walkthrough(env)

    agent = CompressiveTextAgent(max_steps=len(walkthrough) * 2 + 20)
    records: list[dict] = []

    reward, done, step_info = 0.0, False, {}
    just_compressed = False
    action_idx = 0
    guard = 0
    while not done and action_idx < len(walkthrough) and guard < len(walkthrough) * 3:
        guard += 1
        agent.update_from_env(obs, reward, done, step_info)

        total_tokens = sum(agent._token_counter(m["content"]) for m in agent._messages)  # type: ignore[attr-defined]
        should_compress = total_tokens > SFT_SUMMARY_TRIGGER_THRESHOLD and not just_compressed

        prefix = agent.chat_completions  # snapshot BEFORE the synthetic assistant turn

        if should_compress:
            summary_text = extractive_summary(agent._messages, SUMMARY_TOKEN_CAP, agent._token_counter)  # type: ignore[attr-defined]
            target = f"<summary>\n{summary_text}\n</summary>\n\\boxed{{<summarise/>}}"
            is_compression_turn = True
            just_compressed = True
        else:
            oracle_action = walkthrough[action_idx]
            target = f"\\boxed{{{oracle_action}}}"
            is_compression_turn = False
            just_compressed = False
            action_idx += 1

        records.append(
            {
                "messages": prefix,
                "target": target,
                "gamefile": env._gamefile,
                "is_compression_turn": is_compression_turn,
            }
        )

        action = agent.update_from_model(target)
        obs, reward, done, step_info = env.step(action)

    env.close()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-dir", default=str(REPO_ROOT / "games"))
    parser.add_argument("--n-trajectories", type=int, default=500, help="Sec 3.5.2: 500 oracle trajectories.")
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "sft_textworld.jsonl"))
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_records = 0
    with open(out_path, "w") as f:
        for i in range(args.n_trajectories):
            seed = args.seed_start + i
            try:
                records = build_one_trajectory(args.games_dir, seed)
            except Exception as exc:  # noqa: BLE001
                print(f"[traj {i + 1}/{args.n_trajectories}] FAILED (seed={seed}): {exc}")
                continue
            for r in records:
                f.write(json.dumps(r) + "\n")
            n_records += len(records)
            if (i + 1) % 25 == 0:
                print(f"[traj {i + 1}/{args.n_trajectories}] {n_records} SFT examples so far")

    print(f"Done. Wrote {n_records} SFT examples from {args.n_trajectories} trajectories to {out_path}")


if __name__ == "__main__":
    main()
