"""GRPO training entrypoint for the TextWorld context-compression agent.

Mirrors ORBIT's own `scripts/train_multi_episode.py` almost exactly (same
MultiEpisodeEnv wrapper, same `trainers.train_multi_episode.run_ppo_agent` under the
hood) — the only things that change are which agent/env classes get wired in, and
which dataset-prep function builds the task list. We reuse ORBIT's own
`prepare_multi_task_gem_data` unchanged: despite the name, it's fully generic (it
just reads a YAML of {env_id, inner_env_class, env_kwargs...} task specs — see its
docstring) and any extra per-task keys pass straight through into
`inner_env_kwargs["env_kwargs"]`, which is exactly how `TextWorldEnvAdapter` expects
to receive `games_dir` / `split` / `curriculum_enabled` / etc. No need to duplicate it.

Which agent trains depends on `rllm.agent.name` in the Hydra config / CLI override:
  - "compressive_agent"        -> CompressiveTextAgent   (the RL-trained policy, H1)
  - "static_append_agent"      -> HeuristicSummaryAgent(threshold=None) (baseline #1)
  - "heuristic_summary_agent"  -> HeuristicSummaryAgent  (baseline #2 — NOT normally
    trained with GRPO; if you do point GRPO at it, `threshold`/summariser stay fixed,
    so training can only affect *acting*, not compression, which is the ablation
    point of baseline #2's non-learned summariser)

For the "No-RL (SFT-only)" ablation (Sec 3.8, #7): just don't run this script — eval
the SFT checkpoint from `scripts/sft_warmstart.py` directly with
`scripts/eval_textworld_compression.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ORBIT_ROOT = REPO_ROOT.parent  # see README: this file is meant to be dropped into
# an ORBIT checkout and run from ORBIT's own root, so ORBIT's own `data/`, `envs/`,
# `agents/`, `trainers/` packages resolve too. Add both to sys.path defensively.
if str(ORBIT_ROOT) not in sys.path:
    sys.path.insert(0, str(ORBIT_ROOT))

from agents.compressive_text_agent import CompressiveTextAgent  # noqa: E402
from agents.heuristic_summary_agent import HeuristicSummaryAgent  # noqa: E402
from data.prepare_gem_data import prepare_multi_task_gem_data  # noqa: E402  (ORBIT's, reused as-is)
from envs.multi_episode_env import MultiEpisodeEnv  # noqa: E402  (ORBIT's, reused as-is)
from rllm.data import DatasetRegistry  # type: ignore  # noqa: E402
from trainers.train_multi_episode import run_ppo_agent  # noqa: E402  (ORBIT's, reused as-is)


def _static_append_agent(*args, **kwargs) -> HeuristicSummaryAgent:
    kwargs["threshold"] = None
    return HeuristicSummaryAgent(*args, **kwargs)


AGENT_CLASSES = {
    "compressive_agent": CompressiveTextAgent,
    "heuristic_summary_agent": HeuristicSummaryAgent,
    "static_append_agent": _static_append_agent,
}


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(cfg) -> None:  # type: ignore
    tasks_config_path: Optional[str] = cfg.data.get("tasks_config_path", None)
    if not tasks_config_path:
        tasks_config_path = str(REPO_ROOT / "configs" / "textworld_compression_config.yaml")

    tasks_config_path = str(Path(tasks_config_path).expanduser().resolve())
    train_dataset, val_dataset = prepare_multi_task_gem_data(tasks_config_path=tasks_config_path)

    import yaml

    with open(tasks_config_path, "r") as f:
        config = yaml.safe_load(f)
    all_tasks = config.get("train_tasks", []) + config.get("val_tasks", [])
    max_total_step_cap = max((task.get("total_step_cap", 40) for task in all_tasks), default=40)
    if not hasattr(cfg.rllm.agent, "max_steps") or cfg.rllm.agent.max_steps < max_total_step_cap:
        cfg.rllm.agent.max_steps = max_total_step_cap

    if train_dataset is not None:
        cfg.data.train_files = train_dataset.get_verl_data_path()
    if val_dataset is not None:
        cfg.data.val_files = val_dataset.get_verl_data_path()

    agent_args = OmegaConf.to_container(cfg.rllm.agent.get("agent_args", {}), resolve=True)  # type: ignore
    agent_args = dict(agent_args or {})

    agent_name = cfg.rllm.agent.get("name", "compressive_agent")
    agent_class = AGENT_CLASSES.get(agent_name)
    if agent_class is None:
        raise ValueError(f"Unknown rllm.agent.name={agent_name!r}. Choose from {list(AGENT_CLASSES)}.")

    run_ppo_agent(cfg, env_class=MultiEpisodeEnv, agent_class=agent_class, agent_args=agent_args)


if __name__ == "__main__":
    main()
