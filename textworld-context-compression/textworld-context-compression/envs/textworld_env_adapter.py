"""Raw-TextWorld adapter for ORBIT's ``BaseEnv``/``MultiEpisodeEnv`` machinery.

This is the environment half of the proposal's dual system: it plays raw TextWorld
games (NOT ALFWorld's household domain — see proposal Sec 3, which explicitly chooses
TextWorld over the ORBIT/ALFWorld environments for flexibility) and unions the
TextWorld admissible-command action space with the special ``<summarise/>`` action
(Sec 3.3's POMDP formalisation: A_t = A_{s_t} ∪ {<summarise/>}).

Design decisions worth flagging explicitly (these are judgment calls the proposal's
prose underspecifies — sanity-check them against what you and your supervisors
actually intend before a long training run):

1. ``<summarise/>`` is a MUTUALLY EXCLUSIVE per-turn choice, not something bolted onto
   a real action in the same turn. Choosing it consumes one turn of the turn budget
   (`max_turns` / `total_step_cap`) but does NOT change TextWorld's game state — the
   inner environment is not stepped. This matches Fig 3.1 (Action Output -> single
   Environment Step loop) and Sec 3.3's action-space union. The alternative reading
   — compress-then-act in one turn — would need a different parser; see
   ``agents/compressive_text_agent.py`` docstring for the same note.

2. The ``<summary>...</summary>`` CONTENT is produced and consumed entirely on the
   agent side (``CompressiveTextAgent`` edits its own chat history). This env only
   needs to recognise that a compression turn happened — it doesn't parse or store
   summary text. That keeps this class usable by heuristic/baseline agents too.

3. Games are NOT generated at runtime. ``data/generate_textworld_games.py`` builds a
   pool of compiled game files once (with a JSON sidecar of difficulty metadata: map
   size, lock-key depth, distractor count, room-adjacency graph, goal room — needed
   for the curriculum's potential-based shaping). This mirrors ALFWorldEnvAdapter's
   own pattern (pre-built game files, selected deterministically by seed) and avoids
   procedural generation cost inside the training loop.

4. Room-for-shaping is read off TextWorld's ``facts`` extra by scanning for an
   ``at(player, <room>)`` predicate, which is a stable TextWorld convention. If a
   TextWorld version changes this, shaping silently disables for that step rather
   than crashing the rollout — see ``_extract_player_room``.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rllm.agents.agent import Action  # type: ignore
from rllm.environments.base.base_env import BaseEnv  # type: ignore

from envs._admissible_matching import (
    extract_boxed,
    is_summarise_action,
    match_admissible,
)
from envs._tw_threadsafety import TW_REGISTER_LOCK, install_threadlocal_parsers
from envs.curriculum_reward import PotentialShaper

install_threadlocal_parsers()

import textworld  # type: ignore
import textworld.gym  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]

# Class-level cache of (games_dir, split) -> sorted list of (gamefile, metadata) so we
# don't re-glob the filesystem for every parallel rollout worker.
_GAMEFILE_CACHE: Dict[Tuple[str, str], List[Tuple[str, dict]]] = {}
_CACHE_LOCK = threading.Lock()


def _collect_games(games_dir: str, split: str) -> List[Tuple[str, dict]]:
    key = (games_dir, split)
    with _CACHE_LOCK:
        if key in _GAMEFILE_CACHE:
            return _GAMEFILE_CACHE[key]

    split_dir = Path(games_dir) / split
    if not split_dir.is_dir():
        raise RuntimeError(
            f"No games directory at {split_dir}. Run "
            f"`python data/generate_textworld_games.py` first (see README)."
        )

    games: List[Tuple[str, dict]] = []
    for gamefile in sorted(split_dir.glob("*.z8")) + sorted(split_dir.glob("*.ulx")):
        meta_path = gamefile.with_suffix(".json")
        metadata = {}
        if meta_path.is_file():
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        games.append((str(gamefile), metadata))

    if not games:
        raise RuntimeError(
            f"No .z8/.ulx game files found under {split_dir}. Run "
            f"`python data/generate_textworld_games.py` first (see README)."
        )

    with _CACHE_LOCK:
        _GAMEFILE_CACHE[key] = games
    return games


class TextWorldEnvAdapter(BaseEnv):
    """Adapter exposing raw TextWorld games via the ``BaseEnv`` contract.

    Each instance manages a single TextWorld game, selected deterministically from
    `seed` within the requested `split`'s game pool (same pattern as
    ``ALFWorldEnvAdapter``: same seed -> same game -> multi-episode replay works for
    free once wrapped in ``MultiEpisodeEnv``).
    """

    def __init__(
        self,
        env_id: str = "textworld",
        env_kwargs: Optional[Dict[str, Any]] = None,
        max_turns: int = 40,
        seed: Optional[int] = None,
        split: str = "train_nav",
        games_dir: Optional[str] = None,
        count_summarise_in_turns: bool = True,
        curriculum_enabled: bool = False,
        curriculum_alpha0: float = 1.0,
        curriculum_kappa: float = 0.05,
        curriculum_gamma: float = 0.99,
        **_: Any,
    ) -> None:
        super().__init__()

        merged = dict(env_kwargs or {})
        self.env_id = str(merged.get("env_id", env_id))
        self.max_turns = int(merged.get("max_turns", max_turns))
        self._seed = merged.get("seed", seed)
        self.split = str(merged.get("split", split))
        games_dir = merged.get("games_dir", games_dir)
        if games_dir is None:
            games_dir = os.environ.get("TW_GAMES_DIR", str(REPO_ROOT / "games"))
        if not os.path.isabs(games_dir):
            games_dir = str(REPO_ROOT / games_dir)
        self.games_dir = games_dir

        self.count_summarise_in_turns = bool(
            merged.get("count_summarise_in_turns", count_summarise_in_turns)
        )

        self._shaper: Optional[PotentialShaper] = None
        if bool(merged.get("curriculum_enabled", curriculum_enabled)):
            self._shaper = PotentialShaper(
                alpha_0=float(merged.get("curriculum_alpha0", curriculum_alpha0)),
                kappa=float(merged.get("curriculum_kappa", curriculum_kappa)),
                enabled=True,
            )
        self._curriculum_gamma = float(merged.get("curriculum_gamma", curriculum_gamma))
        # Set externally by the trainer/curriculum callback each epoch. Kept as plain
        # state (not threaded through step()) so existing MultiEpisodeEnv call sites
        # don't need to change.
        self.current_epoch: float = 0.0

        self._games = _collect_games(self.games_dir, self.split)

        # Deferred state — created on first reset()
        self._tw_env = None
        self._gamefile: Optional[str] = None
        self._metadata: dict = {}
        self._admissible_commands: List[str] = []
        self._task_description: str = ""
        self.turn: int = 0
        self._done: bool = False
        self._room_before: Optional[str] = None
        self._consecutive_summarise: int = 0
        # Safety valve: refuse to let the policy spam <summarise/> forever without
        # ever taking a real action (that would starve the trajectory of reward
        # signal and burn the whole turn budget). Not in the proposal explicitly;
        # added so a degenerate policy can't stall training. Tune/remove as needed.
        self.max_consecutive_summarise = int(merged.get("max_consecutive_summarise", 3))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, task: Optional[dict] = None) -> Tuple[str, dict]:
        config = self._resolve_reset_config(seed=seed, task=task)
        self._seed = int(config["seed"])
        self.max_turns = int(config["max_turns"])

        gamefile, metadata = self._select_game(self._seed)

        if self._gamefile != gamefile or self._tw_env is None:
            self._init_tw_env(gamefile)
            self._gamefile = gamefile
            self._metadata = metadata
            if self._shaper is not None and metadata.get("room_adjacency"):
                self._shaper.set_graph(
                    {k: v for k, v in metadata["room_adjacency"].items()},
                    goal_room=metadata.get("goal_room"),
                )

        with TW_REGISTER_LOCK:
            obs_list, infos = self._tw_env.reset()
        raw_obs = obs_list[0]
        info = self._unpack_info(infos)

        self._admissible_commands = [c for c in info.get("admissible_commands", []) if c != "help"]
        self._task_description = self._metadata.get("task_description") or self._extract_task(raw_obs)
        self.turn = 0
        self._done = False
        self._consecutive_summarise = 0
        self._room_before = self._extract_player_room(info)

        normalized_info = {
            "env_id": self.env_id,
            "turn": 0,
            "max_turns": self.max_turns,
            "terminated": False,
            "truncated": False,
            "raw_reward": 0.0,
            "gamefile": gamefile,
            "task_type": self._metadata.get("split", self.split),
            "difficulty": self._metadata.get("difficulty", {}),
        }
        return self._format_init_obs(raw_obs), normalized_info

    def step(self, action: Any) -> Tuple[str, float, bool, dict]:
        if self._done:
            raise RuntimeError("Environment is done. Call reset() before step().")

        parsed = self._parse_action(action)

        if is_summarise_action(parsed):
            return self._step_compress()

        self._consecutive_summarise = 0
        matched = match_admissible(parsed, self._admissible_commands)

        with TW_REGISTER_LOCK:
            pass  # step() itself doesn't need the registration lock (parsers are thread-local)
        obs_list, scores, dones, infos = self._tw_env.step([matched])
        raw_obs = obs_list[0]
        info = self._unpack_info(infos)

        self.turn += 1
        self._admissible_commands = [c for c in info.get("admissible_commands", []) if c != "help"]

        won = bool(info.get("won", False))
        lost = bool(info.get("lost", False))
        inner_done = bool(dones[0]) if hasattr(dones, "__getitem__") else bool(dones)
        terminated = won or lost or inner_done
        truncated = not terminated and self.turn >= self.max_turns
        done = terminated or truncated
        self._done = done

        native_reward = 1.0 if won else 0.0

        room_after = self._extract_player_room(info)
        shaped_reward = native_reward
        if self._shaper is not None:
            shaped_reward = self._shaper.shape(
                native_reward,
                self._room_before,
                room_after,
                epoch=self.current_epoch,
                gamma=self._curriculum_gamma,
                terminal=terminated,
            )
        self._room_before = room_after

        normalized_info = {
            "env_id": self.env_id,
            "turn": self.turn,
            "max_turns": self.max_turns,
            "terminated": bool(terminated),
            "truncated": truncated,
            "raw_reward": native_reward,
            "shaped_reward": shaped_reward,
            "is_correct": won,
            "parsed_action": matched,
            "was_compression_turn": False,
            "gamefile": self._gamefile,
        }

        obs_text = self._format_step_obs(raw_obs, won, lost, terminated, truncated)
        return obs_text, shaped_reward, done, normalized_info

    def close(self) -> None:
        if self._tw_env is not None:
            try:
                self._tw_env.close()
            except Exception:
                pass
            self._tw_env = None

    @staticmethod
    def from_dict(info: dict) -> "TextWorldEnvAdapter":
        env_kwargs = dict(info.get("env_kwargs", {}) or {})
        passthrough_keys = (
            "max_turns",
            "seed",
            "split",
            "games_dir",
            "count_summarise_in_turns",
            "curriculum_enabled",
            "curriculum_alpha0",
            "curriculum_kappa",
            "curriculum_gamma",
            "max_consecutive_summarise",
        )
        for key in passthrough_keys:
            if key in info and key not in env_kwargs:
                env_kwargs[key] = info[key]
        return TextWorldEnvAdapter(env_id=info.get("env_id", "textworld"), env_kwargs=env_kwargs)

    @staticmethod
    def is_multithread_safe() -> bool:
        # Same reasoning as ALFWorldEnvAdapter: game registration/loading routes
        # through a global lock, so treat instances as not safe to share across
        # threads (each rollout worker should own its own instance).
        return False

    # ------------------------------------------------------------------
    # Compression turn handling
    # ------------------------------------------------------------------

    def _step_compress(self) -> Tuple[str, float, bool, dict]:
        """Handle a <summarise/> turn: no-op w.r.t. game state, consumes (by default)
        one turn of budget. See module docstring point (1) for the design rationale.
        """
        self._consecutive_summarise += 1
        if self.count_summarise_in_turns:
            self.turn += 1

        truncated = self.count_summarise_in_turns and self.turn >= self.max_turns
        # Forced-compliance backstop: if the policy tries to summarise more than
        # `max_consecutive_summarise` times in a row without acting, stop honouring
        # <summarise/> and nudge it to act instead, so the trajectory can't stall
        # indefinitely just accumulating shaped_reward=0 turns.
        force_act_notice = ""
        if self._consecutive_summarise > self.max_consecutive_summarise:
            force_act_notice = (
                "\n\n[You have summarised several turns in a row. "
                "Choose a real admissible action now.]"
            )

        done = truncated
        self._done = done

        admissible_str = ", ".join(f"'{cmd}'" for cmd in self._admissible_commands)
        obs_text = (
            "[Context compressed.]\n\n"
            f"Admissible actions: [{admissible_str}]{force_act_notice}\n\n"
            "Choose one admissible action and output it inside \\boxed{}."
        )
        normalized_info = {
            "env_id": self.env_id,
            "turn": self.turn,
            "max_turns": self.max_turns,
            "terminated": False,
            "truncated": truncated,
            "raw_reward": 0.0,
            "shaped_reward": 0.0,
            "is_correct": False,
            "parsed_action": "<summarise/>",
            "was_compression_turn": True,
            "gamefile": self._gamefile,
        }
        return obs_text, 0.0, done, normalized_info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_game(self, seed: int) -> Tuple[str, dict]:
        idx = seed % len(self._games)
        return self._games[idx]

    def _init_tw_env(self, gamefile: str) -> None:
        if self._tw_env is not None:
            try:
                self._tw_env.close()
            except Exception:
                pass

        request_infos = textworld.EnvInfos(
            won=True,
            lost=True,
            admissible_commands=True,
            facts=True,  # needed for `_extract_player_room` (curriculum shaping)
            extras=["gamefile"],
        )
        with TW_REGISTER_LOCK:
            env_id = textworld.gym.register_games(
                [gamefile],
                request_infos,
                batch_size=1,
                asynchronous=False,
                max_episode_steps=self.max_turns * 4,  # generous; our own max_turns governs truncation
            )
            self._tw_env = textworld.gym.make(env_id)

    def _resolve_reset_config(self, seed: Optional[int], task: Optional[dict]) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "seed": self._seed if self._seed is not None else 0,
            "max_turns": self.max_turns,
        }
        if isinstance(task, dict):
            if "seed" in task and task["seed"] is not None:
                config["seed"] = int(task["seed"])
            if "max_turns" in task and task["max_turns"] is not None:
                config["max_turns"] = int(task["max_turns"])
            elif "max_turns_per_episode" in task and task["max_turns_per_episode"] is not None:
                config["max_turns"] = int(task["max_turns_per_episode"])
        if seed is not None:
            config["seed"] = int(seed)
        return config

    @staticmethod
    def _unpack_info(infos: dict) -> dict:
        out: Dict[str, Any] = {}
        for k, v in infos.items():
            if isinstance(v, (list, tuple)) and len(v) == 1:
                out[k] = v[0]
            else:
                out[k] = v
        return out

    def _parse_action(self, action: Any) -> str:
        if isinstance(action, Action):
            action = action.action
        return extract_boxed(str(action).strip())

    def _extract_player_room(self, info: dict) -> Optional[str]:
        """Scan TextWorld `facts` for an `at(player, <room>)` predicate.

        Defensive by design: curriculum shaping is a training-time nicety, not
        something that should ever crash a rollout. Any parsing failure here just
        disables shaping for this step (falls back to native reward).
        """
        if self._shaper is None:
            return None
        facts = info.get("facts")
        if not facts:
            return None
        try:
            for fact in facts:
                name = getattr(fact, "name", None)
                args = getattr(fact, "arguments", None) or getattr(fact, "args", None)
                if name != "at" or not args or len(args) < 2:
                    continue
                first_name = getattr(args[0], "name", str(args[0]))
                if first_name != "player":
                    continue
                return getattr(args[1], "name", str(args[1]))
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    # Observation formatting (ORBIT style, matches ALFWorldEnvAdapter)
    # ------------------------------------------------------------------

    def _format_init_obs(self, raw_obs: str) -> str:
        admissible_str = ", ".join(f"'{cmd}'" for cmd in self._admissible_commands)
        task_line = f"Your task is to: {self._task_description}\n\n" if self._task_description else ""
        return (
            "You are in a text-based environment.\n"
            f"{task_line}"
            f"{raw_obs}\n\n"
            f"Admissible actions: [{admissible_str}]\n\n"
            "Choose one admissible action and output it inside \\boxed{}. "
            "If your context is getting long, you may instead output "
            "\\boxed{<summarise/>} to compress your history — see the system prompt "
            "for the summary format."
        )

    def _format_step_obs(self, raw_obs: str, won: bool, lost: bool, terminated: bool, truncated: bool) -> str:
        if won:
            return f"Observation: {raw_obs}\nCongratulations! You have completed the task successfully."
        if lost:
            return f"Observation: {raw_obs}\nThe episode ended in failure."
        if terminated and not truncated:
            return f"Observation: {raw_obs}\nEpisode finished. The task was not completed."
        if truncated:
            return f"Observation: {raw_obs}\nEpisode stopped because max_turns ({self.max_turns}) was reached."
        admissible_str = ", ".join(f"'{cmd}'" for cmd in self._admissible_commands)
        return (
            f"Observation: {raw_obs}\n\n"
            f"Admissible actions: [{admissible_str}]\n\n"
            "Choose one admissible action and output it inside \\boxed{}, "
            "or \\boxed{<summarise/>} to compress."
        )

    @staticmethod
    def _extract_task(text_obs: str) -> str:
        marker = "Your task is to: "
        idx = text_obs.find(marker)
        if idx == -1:
            return ""
        rest = text_obs[idx + len(marker):]
        newline = rest.find("\n")
        return (rest[:newline] if newline != -1 else rest).strip()
