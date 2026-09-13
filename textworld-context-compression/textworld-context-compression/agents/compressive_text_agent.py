"""The core novel piece: a GRPO-trained agent that manages its own context via
periodic <summary> blocks (proposal Sections 3.5.3, 3.3).

How this fits into the rLLM turn loop (see ``rllm/agents/agent.py::BaseAgent``):

    engine loop, each turn:
        response = model(agent.chat_completions)
        action   = agent.update_from_model(response)      # <- we parse here
        obs, reward, done, info = env.step(action)         # <- TextWorldEnvAdapter
        agent.update_from_env(obs, reward, done, info)      # <- we append/compress here

The COMPRESSION SURGERY happens in `update_from_model`, the moment we know the model
chose to summarise: we splice `self._messages` down to
``[system_prompt, compressed_block]`` right then, so that the very next
`update_from_env` call appends the post-compression observation onto an already-short
history. This is what makes "the compressed history h'_t is the compressed history
available in the context window" (Sec 3.3) literally true turn-by-turn, not just at
episode boundaries.

Design note (see also `envs/textworld_env_adapter.py` docstring point 1): a
compression turn is DECLARED by the model in the response passed to
`update_from_model` ("did this response request `<summarise/>`?"), and it is
CONSUMED by the environment as one no-op turn. The agent doesn't need to talk to the
environment to know a compression happened — it already parsed the same response the
env will parse. We re-parse independently (deliberately) rather than trust the env's
`info["was_compression_turn"]`, so this agent also works standalone (e.g. in the SFT
data generator, without any env in the loop at all).

Two independent triggers can cause a compression:
  1. Policy-chosen: the model itself emits \\boxed{<summarise/>} plus a <summary> block.
  2. Forced fallback (Sec 3.5.3): accumulated context exceeds
     `force_summary_token_threshold` (default 1024) regardless of what the model did.
     This is a DETERMINISTIC safety net, not a soft nudge — if the model didn't
     summarise and didn't act within budget, we synthesize a crude extractive summary
     ourselves (`_fallback_summary`) so context never silently blows past the cap.
     This mirrors the SUPO-style thresholding the proposal cites as precedent.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Callable, List, Optional

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory  # type: ignore

from agents._extractive_summary import extractive_summary, truncate_to_tokens
from envs._admissible_matching import (
    extract_boxed,
    extract_summary_block,
    is_summarise_action,
)
from prompts.compression_prompts import (
    FORCE_SUMMARY_TOKEN_THRESHOLD,
    SUMMARY_TOKEN_CAP,
    SYSTEM_PROMPT_COMPRESSIVE,
)

COMPRESS_SENTINEL = "<summarise/>"


def _default_token_counter(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 chars/token for English).

    Good enough for threshold decisions during rollout; pass a real tokenizer's
    `.encode` (via `token_counter=lambda t: len(tok.encode(t))`) for anything you
    report numbers from (e.g. the RQ2/H2 style token-budget claims — though those
    were part of the KV-quantisation layer you've cut, the same swap-in point applies
    if you ever want exact counts here too).
    """
    return max(1, len(text) // 4)


class CompressiveTextAgent(BaseAgent):
    """GRPO-trained agent implementing periodic-summary context compression."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_steps: int = 40,
        summary_token_cap: int = SUMMARY_TOKEN_CAP,
        force_summary_token_threshold: int = FORCE_SUMMARY_TOKEN_THRESHOLD,
        token_counter: Optional[Callable[[str], int]] = None,
    ):
        self.system_prompt = system_prompt or SYSTEM_PROMPT_COMPRESSIVE
        self.max_steps = max_steps
        self.summary_token_cap = summary_token_cap
        self.force_summary_token_threshold = force_summary_token_threshold
        self._token_counter = token_counter or _default_token_counter

        self._messages: List[dict] = []
        self._trajectory = Trajectory()
        # Index into self._messages right after the system prompt where the current
        # "compressed block" (a single user-role message) lives, or None if no
        # compression has happened yet this episode/trajectory.
        self._compressed_block_idx: Optional[int] = None
        self._last_compression_reason: Optional[str] = None
        self.reset()

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        return copy.deepcopy(self._messages)

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def reset(self):
        self._messages = [{"role": "system", "content": self.system_prompt}]
        self._trajectory = Trajectory()
        self._compressed_block_idx = None
        self._last_compression_reason = None

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        self._messages.append({"role": "user", "content": str(observation)})
        self._maybe_force_summary()

        if self._trajectory.steps:
            last_step = self._trajectory.steps[-1]
            last_step.reward = float(reward)
            last_step.done = bool(done)
            last_step.info.update(info or {})

    def update_from_model(self, response: str, **kwargs) -> Action:
        parsed_action = extract_boxed(response)
        self._messages.append({"role": "assistant", "content": response})

        if is_summarise_action(parsed_action):
            summary_text = extract_summary_block(response)
            if summary_text is None:
                # Model claimed <summarise/> but didn't produce a parseable
                # <summary> block. Don't silently drop history in that case — fall
                # back to treating this as a malformed turn: compress using a crude
                # truncation of the raw history instead, and flag it in info so
                # RQ1-style malformed-output tracking can see it.
                summary_text = self._fallback_summary(reason="malformed_summarise")
                malformed = True
            else:
                summary_text = self._truncate_to_tokens(summary_text, self.summary_token_cap)
                malformed = False

            self._apply_compression(summary_text)
            action = Action(action=COMPRESS_SENTINEL)
            step_info = {"compression_reason": "policy", "malformed_summary_tag": malformed}
        else:
            action = Action(action=parsed_action)
            step_info = {}

        step = Step(
            chat_completions=copy.deepcopy(self._messages),
            observation=self._messages[-2]["content"] if len(self._messages) >= 2 else None,
            action=action,
            model_response=response,
            info=step_info,
        )
        self._trajectory.steps.append(step)
        return action

    def get_current_state(self) -> Step:
        if not self._trajectory.steps:
            return Step(chat_completions=copy.deepcopy(self._messages))
        return self._trajectory.steps[-1]

    # ------------------------------------------------------------------
    # Compression internals
    # ------------------------------------------------------------------

    def _apply_compression(self, summary_text: str) -> None:
        """Splice self._messages down to [system, compressed_block].

        Any raw turns since the last compression (or since episode start) are
        discarded — that IS the compression. The compressed block is a single
        user-role message so it reads naturally as context for the next real
        action, and is styled distinctly ("[COMPRESSED HISTORY]") so it's easy to
        `grep` trajectories for compression events during debugging/eval.
        """
        block = {"role": "user", "content": f"[COMPRESSED HISTORY]\n{summary_text}"}
        self._messages = [self._messages[0], block]
        self._compressed_block_idx = 1

    def _maybe_force_summary(self) -> None:
        """Deterministic fallback (Sec 3.5.3): if context has grown past
        `force_summary_token_threshold` since the last compression, compress now
        using a crude extractive summary of the raw turns, without waiting for the
        model to choose to. Runs right after appending the newest observation, so
        the NEXT model call always sees a context under budget.
        """
        total_tokens = sum(self._token_counter(m["content"]) for m in self._messages)
        if total_tokens <= self.force_summary_token_threshold:
            return
        summary_text = self._fallback_summary(reason="forced_threshold")
        self._apply_compression(summary_text)
        # Re-append: _apply_compression truncated away the observation we just
        # added, so put a short pointer back so the model isn't left staring at
        # only a summary with no indication of what it should do this turn. The
        # real, full observation text got folded into the fallback summary itself
        # via `_fallback_summary`, so this is just a "continue" nudge.
        self._messages.append(
            {
                "role": "user",
                "content": (
                    "[Context force-compressed — you did not summarise before the "
                    f"{self.force_summary_token_threshold}-token budget. See the "
                    "compressed history above, then continue with your next action.]"
                ),
            }
        )

    def _fallback_summary(self, reason: str) -> str:
        """Crude, non-learned extractive summary used only as a deterministic
        safety net (never by the trained policy on a normal turn). Concatenates the
        last few raw user/assistant turns, truncated to the token cap. This is
        intentionally dumb — if it's firing often during eval, that's a sign the
        policy isn't summarising proactively enough (worth logging as a metric).
        """
        self._last_compression_reason = reason
        return extractive_summary(self._messages, self.summary_token_cap, self._token_counter)

    def _truncate_to_tokens(self, text: str, token_cap: int) -> str:
        return truncate_to_tokens(text, token_cap, self._token_counter)
