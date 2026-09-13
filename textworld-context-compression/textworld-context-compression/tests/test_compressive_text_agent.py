"""Pure-Python unit tests for `CompressiveTextAgent`'s message-splicing logic — no
TextWorld/rLLM-rollout dependency needed beyond the `rllm.agents.agent` dataclasses,
so this runs anywhere `third_party/rllm` is importable (even without vLLM/verl/GPU).

Run with: `python -m pytest tests/test_compressive_text_agent.py -v`
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ORBIT_ROOT = REPO_ROOT.parent
for p in (REPO_ROOT, ORBIT_ROOT, ORBIT_ROOT / "third_party" / "rllm"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest  # noqa: E402

from agents.compressive_text_agent import COMPRESS_SENTINEL, CompressiveTextAgent  # noqa: E402


def test_real_action_turn_appends_normally():
    agent = CompressiveTextAgent()
    agent.update_from_env("You are in a room.", 0.0, False, {})
    action = agent.update_from_model(r"I should look around. \boxed{go north}")
    assert action.action == "go north"
    assert agent._messages[-1]["role"] == "assistant"
    assert len(agent._messages) == 3  # system, user(obs), assistant


def test_compression_turn_splices_history():
    agent = CompressiveTextAgent()
    agent.update_from_env("obs 1", 0.0, False, {})
    agent.update_from_model(r"\boxed{go north}")
    agent.update_from_env("obs 2", 0.0, False, {})
    agent.update_from_model(r"\boxed{take key}")
    agent.update_from_env("obs 3", 0.0, False, {})

    response = r"<summary>Found a key in room 1, heading north.</summary>\boxed{<summarise/>}"
    action = agent.update_from_model(response)

    assert action.action == COMPRESS_SENTINEL
    # messages should now be exactly [system, compressed_block]
    assert len(agent._messages) == 2
    assert agent._messages[0]["role"] == "system"
    assert "[COMPRESSED HISTORY]" in agent._messages[1]["content"]
    assert "Found a key" in agent._messages[1]["content"]


def test_compression_then_next_observation_appends_after_summary():
    agent = CompressiveTextAgent()
    agent.update_from_env("obs 1", 0.0, False, {})
    agent.update_from_model(r"<summary>nothing yet</summary>\boxed{<summarise/>}")
    agent.update_from_env("obs after compression", 0.0, False, {})

    assert len(agent._messages) == 3  # system, compressed_block, new observation
    assert agent._messages[-1]["content"] == "obs after compression"


def test_malformed_summarise_falls_back_without_crashing():
    agent = CompressiveTextAgent()
    agent.update_from_env("obs 1", 0.0, False, {})
    # Claims <summarise/> but never produces a <summary> block.
    action = agent.update_from_model(r"\boxed{<summarise/>}")
    assert action.action == COMPRESS_SENTINEL
    step = agent.get_current_state()
    assert step.info.get("malformed_summary_tag") is True
    # Should still have compressed down to [system, fallback_block] rather than
    # leaving stale raw history around.
    assert len(agent._messages) == 2


def test_forced_summary_triggers_past_token_threshold():
    agent = CompressiveTextAgent(force_summary_token_threshold=20, token_counter=lambda t: len(t.split()))
    long_obs = " ".join(f"word{i}" for i in range(100))
    agent.update_from_env(long_obs, 0.0, False, {})
    # Should have force-compressed already inside update_from_env.
    assert len(agent._messages) <= 3  # system, compressed_block, continue-nudge
    assert any("force-compressed" in m["content"] for m in agent._messages if m["role"] == "user")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
