"""Baseline #2 from the proposal (Sec 3.8): "a model that triggers a post-hoc
generative summarisation prompt when the context window fills, mirroring procedural
memory baselines like MemP. Isolates whether GRPO retains more useful information
than a non-learned summariser."

Key contrast with `CompressiveTextAgent`: the model here never sees `<summarise/>` in
its action space and is never trained to decide when/what to compress. Compression is
entirely scripted — triggered purely by a token-count threshold, using the same
deterministic extractive summariser as `CompressiveTextAgent`'s *forced fallback* path
(see `agents/_extractive_summary.py`). That's the controlled variable: both baselines
end up using identical non-learned summarisation content when they fall back to it;
the only thing GRPO training can improve is picking better moments/content
*proactively*, which this baseline can't do by construction.

Also serves double duty as the "Static-Append" baseline's cousin: set
`threshold=None` to disable compression entirely and get plain static-append
behaviour with TextWorld-appropriate prompting (equivalent to reusing ORBIT's
`GEMTextAgent` with `SYSTEM_PROMPT_STATIC`, but self-contained here so this file
doesn't need ALFWorld/GEM-specific imports).
"""

from __future__ import annotations

import copy
from typing import Any, Callable, List, Optional

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory  # type: ignore

from agents._extractive_summary import extractive_summary
from envs._admissible_matching import extract_boxed
from prompts.compression_prompts import FORCE_SUMMARY_TOKEN_THRESHOLD, SUMMARY_TOKEN_CAP, SYSTEM_PROMPT_STATIC


def _default_token_counter(text: str) -> int:
    return max(1, len(text) // 4)


class HeuristicSummaryAgent(BaseAgent):
    """Scripted context-length-triggered summarisation, no RL involved."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_steps: int = 40,
        threshold: Optional[int] = FORCE_SUMMARY_TOKEN_THRESHOLD,
        summary_token_cap: int = SUMMARY_TOKEN_CAP,
        token_counter: Optional[Callable[[str], int]] = None,
    ):
        self.system_prompt = system_prompt or SYSTEM_PROMPT_STATIC
        self.max_steps = max_steps
        self.threshold = threshold  # None disables compression -> static-append baseline
        self.summary_token_cap = summary_token_cap
        self._token_counter = token_counter or _default_token_counter

        self._messages: List[dict] = []
        self._trajectory = Trajectory()
        self.reset()

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        return copy.deepcopy(self._messages)

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def reset(self):
        self._messages = [{"role": "system", "content": self.system_prompt}]
        self._trajectory = Trajectory()

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        self._messages.append({"role": "user", "content": str(observation)})
        self._maybe_compress()

        if self._trajectory.steps:
            last_step = self._trajectory.steps[-1]
            last_step.reward = float(reward)
            last_step.done = bool(done)
            last_step.info.update(info or {})

    def update_from_model(self, response: str, **kwargs) -> Action:
        # No <summarise/> in this agent's vocabulary: everything the model outputs
        # is a real action attempt, matched against admissible commands downstream
        # by the env (same as the static-append path).
        parsed_action = extract_boxed(response)
        self._messages.append({"role": "assistant", "content": response})

        step = Step(
            chat_completions=copy.deepcopy(self._messages),
            observation=self._messages[-2]["content"] if len(self._messages) >= 2 else None,
            action=Action(action=parsed_action),
            model_response=response,
            info={},
        )
        self._trajectory.steps.append(step)
        return Action(action=parsed_action)

    def get_current_state(self) -> Step:
        if not self._trajectory.steps:
            return Step(chat_completions=copy.deepcopy(self._messages))
        return self._trajectory.steps[-1]

    # ------------------------------------------------------------------
    def _maybe_compress(self) -> None:
        if self.threshold is None:
            return
        total_tokens = sum(self._token_counter(m["content"]) for m in self._messages)
        if total_tokens <= self.threshold:
            return
        summary_text = extractive_summary(self._messages, self.summary_token_cap, self._token_counter)
        block = {"role": "user", "content": f"[SUMMARISED HISTORY]\n{summary_text}"}
        # Keep the very last message (the observation we just appended) so the model
        # isn't left with only a summary and no indication of the current turn.
        latest = self._messages[-1]
        self._messages = [self._messages[0], block, latest]
