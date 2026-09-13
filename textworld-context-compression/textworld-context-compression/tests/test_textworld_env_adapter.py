"""Tests for TextWorldEnvAdapter, mirroring ORBIT's own `tests/test_alfworld_env_adapter.py`
pattern: pure-unit tests always run; integration tests are skipped unless a real
`textworld` install AND a generated game pool (`games/train_nav/`) are present.

Integration tests need:
    pip install textworld
    python data/generate_textworld_games.py --smoke-test
    TW_GAMES_DIR=games pytest tests/test_textworld_env_adapter.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ORBIT_ROOT = REPO_ROOT.parent
for p in (REPO_ROOT, ORBIT_ROOT, ORBIT_ROOT / "third_party" / "rllm"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_TEXTWORLD_AVAILABLE = False
try:
    from envs.textworld_env_adapter import TextWorldEnvAdapter  # noqa: F401
    _TEXTWORLD_AVAILABLE = True
except Exception:
    TextWorldEnvAdapter = None  # type: ignore[assignment,misc]

_GAMES_DIR = os.environ.get("TW_GAMES_DIR", str(REPO_ROOT / "games" / "_smoke_test"))
_HAS_GAMES = Path(_GAMES_DIR, "train_nav").is_dir()

needs_textworld = pytest.mark.skipif(not _TEXTWORLD_AVAILABLE, reason="textworld not importable")
needs_games = pytest.mark.skipif(
    not (_TEXTWORLD_AVAILABLE and _HAS_GAMES),
    reason=f"No generated games at {_GAMES_DIR}/train_nav — run data/generate_textworld_games.py --smoke-test first",
)


# ======================================================================
# Pure-unit tests (no textworld install needed)
# ======================================================================


class TestAdmissibleMatchingUtils:
    def test_extract_boxed(self):
        from envs._admissible_matching import extract_boxed

        assert extract_boxed(r"reasoning... \boxed{go north}") == "go north"
        assert extract_boxed("plain text") == "plain text"

    def test_is_summarise_action_variants(self):
        from envs._admissible_matching import is_summarise_action

        for text in ["<summarise/>", " <SUMMARISE/> ", "<summarize/>", "summarise"]:
            assert is_summarise_action(text), text
        assert not is_summarise_action("go north")

    def test_match_admissible_exact_and_fuzzy(self):
        from envs._admissible_matching import match_admissible

        admissible = ["go north", "go south", "take brass key"]
        assert match_admissible("go north", admissible) == "go north"
        assert match_admissible("go nort", admissible) == "go north"  # fuzzy
        assert match_admissible("brass key", admissible) == "take brass key"  # substring
        assert match_admissible("", []) == "look"

    def test_extract_summary_block(self):
        from envs._admissible_matching import extract_summary_block

        assert extract_summary_block("<summary>hi</summary>") == "hi"
        assert extract_summary_block("no tags") is None


class TestPotentialShaper:
    def test_shortest_path_potential(self):
        from envs.curriculum_reward import PotentialShaper

        shaper = PotentialShaper(alpha_0=1.0, kappa=0.0)
        shaper.set_graph({"a": ["b"], "b": ["a", "c"], "c": ["b"]}, goal_room="c")
        assert shaper.potential("c") == 0.0
        assert shaper.potential("b") == -1.0
        assert shaper.potential("a") == -2.0

    def test_shaping_rewards_progress(self):
        from envs.curriculum_reward import PotentialShaper

        shaper = PotentialShaper(alpha_0=1.0, kappa=0.0)
        shaper.set_graph({"a": ["b"], "b": ["a", "c"], "c": ["b"]}, goal_room="c")
        shaped = shaper.shape(0.0, room_before="a", room_after="b", epoch=0, gamma=1.0)
        assert shaped > 0

    def test_decay_toward_zero(self):
        from envs.curriculum_reward import PotentialShaper

        shaper = PotentialShaper(alpha_0=1.0, kappa=1.0)
        assert shaper.alpha(0) == pytest.approx(1.0)
        assert shaper.alpha(50) < 0.01  # should have decayed hard by epoch 50


# ======================================================================
# Integration tests (need `pip install textworld` + generated games)
# ======================================================================


@needs_games
class TestTextWorldEnvAdapterIntegration:
    def test_reset_returns_observation_and_admissible_actions(self):
        env = TextWorldEnvAdapter(env_kwargs={"games_dir": _GAMES_DIR, "split": "train_nav", "seed": 0, "max_turns": 20})
        obs, info = env.reset(seed=0)
        assert isinstance(obs, str) and len(obs) > 0
        assert "Admissible actions" in obs
        env.close()

    def test_summarise_action_is_a_noop_step(self):
        env = TextWorldEnvAdapter(env_kwargs={"games_dir": _GAMES_DIR, "split": "train_nav", "seed": 0, "max_turns": 20})
        obs1, _ = env.reset(seed=0)
        obs2, reward, done, info = env.step(r"\boxed{<summarise/>}")
        assert reward == 0.0
        assert info["was_compression_turn"] is True
        assert not done
        env.close()

    def test_same_seed_gives_same_game(self):
        env_a = TextWorldEnvAdapter(env_kwargs={"games_dir": _GAMES_DIR, "split": "train_nav", "seed": 3, "max_turns": 20})
        env_b = TextWorldEnvAdapter(env_kwargs={"games_dir": _GAMES_DIR, "split": "train_nav", "seed": 3, "max_turns": 20})
        obs_a, _ = env_a.reset(seed=3)
        obs_b, _ = env_b.reset(seed=3)
        assert obs_a == obs_b
        env_a.close()
        env_b.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
