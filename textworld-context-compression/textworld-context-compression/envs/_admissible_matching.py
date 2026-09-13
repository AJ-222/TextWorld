"""Shared action-parsing helpers, mirroring ORBIT's ``ALFWorldEnvAdapter``.

Factored out so both the ALFWorld adapter (in ORBIT proper) and our new
``TextWorldEnvAdapter`` use the same \\boxed{} extraction + fuzzy admissible-command
matching, instead of copy-pasting ~40 lines. We deliberately do NOT import from
``envs.alfworld_env_adapter`` (that module has a hard import-time dependency on the
``alfworld`` package via ``_setup_alfworld_imports()``, which we don't want to force
on people who only care about raw TextWorld).
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Optional

BOXED_PATTERN = re.compile(r"\\boxed\{([^}]+)\}", re.IGNORECASE)

# The special compression action. Case/whitespace-insensitive on the way in
# (see `is_summarise_action`), but this is the canonical spelling we emit and
# document everywhere (prompts, SFT data, README).
SUMMARISE_ACTION = "<summarise/>"


def extract_boxed(text: str) -> str:
    """Extract the last ``\\boxed{...}`` substring; fall back to the raw (stripped)
    text if no box is found, so a model that forgets the box format doesn't just
    silently no-op."""
    matches = list(BOXED_PATTERN.finditer(text))
    if not matches:
        return text.strip()
    return matches[-1].group(1).strip()


def is_summarise_action(parsed: str) -> bool:
    """Whitespace/case-insensitive check for the compression sentinel."""
    normalized = re.sub(r"\s+", "", parsed).lower()
    return normalized in ("<summarise/>", "<summarize/>", "summarise", "summarize")


def match_admissible(parsed: str, admissible: List[str]) -> str:
    """Resolve free-form model text to one of the environment's admissible commands.

    Strategy (identical to ``ALFWorldEnvAdapter._match_admissible``): exact -> prefix
    -> substring -> fuzzy (SequenceMatcher, threshold 0.6) -> fallback to "look" or
    the first admissible command.
    """
    if not admissible:
        return "look"

    parsed_lower = parsed.lower().strip()

    for cmd in admissible:
        if cmd.lower().strip() == parsed_lower:
            return cmd

    for cmd in admissible:
        if cmd.lower().startswith(parsed_lower):
            return cmd

    for cmd in admissible:
        if parsed_lower in cmd.lower():
            return cmd

    best_ratio, best_cmd = 0.0, None
    for cmd in admissible:
        ratio = SequenceMatcher(None, parsed_lower, cmd.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_cmd = cmd
    if best_ratio > 0.6 and best_cmd is not None:
        return best_cmd

    for cmd in admissible:
        if cmd.lower() == "look":
            return cmd
    return admissible[0]


def extract_summary_block(text: str) -> Optional[str]:
    """Pull the content of a ``<summary>...</summary>`` block out of a model response.

    Returns None if no (closed) summary block is present. Case-insensitive on the
    tags since models are inconsistent about this.
    """
    match = re.search(r"<summary>(.*?)</summary>", text, re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()
