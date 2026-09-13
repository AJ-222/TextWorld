"""Deterministic, non-learned summarisation used by (a) the trained
`CompressiveTextAgent`'s forced-fallback path and (b) the whole
`HeuristicSummaryAgent` baseline. Shared so the two only differ in *when* they
compress, not *how* the non-learned compression itself works — which is the point
of the RQ1/H1 comparison (Section 3.8: does the RL-trained policy pick better
compression content than a fixed heuristic?).
"""

from __future__ import annotations

from typing import Callable, List


def extractive_summary(
    messages: List[dict],
    token_cap: int,
    token_counter: Callable[[str], int],
    tail_turns: int = 6,
) -> str:
    """Concatenate the last `tail_turns` raw user/assistant messages (skipping the
    system prompt), truncated to `token_cap`. Deliberately dumb: no learned
    selection of what matters, just recency.
    """
    raw_turns = [m for m in messages[1:] if m["role"] in ("user", "assistant")]
    tail = raw_turns[-tail_turns:]
    joined = "\n".join(f"[{m['role']}] {m['content']}" for m in tail)
    return truncate_to_tokens(joined, token_cap, token_counter)


def truncate_to_tokens(text: str, token_cap: int, token_counter: Callable[[str], int]) -> str:
    if token_counter(text) <= token_cap:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if token_counter(text[:mid]) <= token_cap:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + " …"
