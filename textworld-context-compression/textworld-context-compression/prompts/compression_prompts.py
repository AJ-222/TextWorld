"""Prompt templates for the compression-policy agent family.

`SYSTEM_PROMPT_COMPRESSIVE` is the system prompt for the GRPO-trained
`CompressiveTextAgent`. `SYSTEM_PROMPT_STATIC` is the plain no-compression baseline
prompt (mirrors ORBIT's `_default_multi_episode_prompt`, adapted for TextWorld).
`HEURISTIC_SUMMARY_INSTRUCTION` is the one-off prompt used by the *scripted* heuristic
baseline (Section 3.8, baseline #2) when it triggers a post-hoc summarisation call —
note this baseline does NOT get GRPO-trained to write good summaries, it just always
asks for one the same way, which is the point of the comparison.
"""

SUMMARY_TOKEN_CAP = 256
FORCE_SUMMARY_TOKEN_THRESHOLD = 1024
SFT_SUMMARY_TRIGGER_THRESHOLD = 512  # Section 3.5.2: SFT warm-start inserts <summarise/>
# whenever accumulated history exceeds this many tokens.

SYSTEM_PROMPT_COMPRESSIVE = f"""You are an agent solving a text-based environment across multiple episodes of the \
same task. You must manage your own context window.

Each turn you choose EXACTLY ONE of:
1. A real admissible action, output as \\boxed{{action text}} (copy one of the \
admissible actions shown to you, exactly).
2. The special compression action \\boxed{{<summarise/>}}. If you choose this, first \
write a summary of everything useful from the interaction so far inside \
<summary>...</summary> tags (max {SUMMARY_TOKEN_CAP} tokens), THEN output \
\\boxed{{<summarise/>}}. Your raw interaction history will be replaced by this \
summary — after this turn you will no longer be able to see the raw history, only \
your summary — so include anything you would regret forgetting: locations of \
objects, keys and their locks, what you have already tried and failed, and what you \
still need to do.

Rules for the summary:
- Do NOT restate the interaction turn-by-turn; write a condensed belief state.
- Separate confirmed facts from things you are still uncertain about.
- Be concrete: name rooms, objects, and unlock dependencies explicitly.
- If your context grows past roughly {FORCE_SUMMARY_TOKEN_THRESHOLD} tokens without \
you choosing to summarise, the system will force a summary for you using a much \
cruder heuristic — you are better off summarising yourself, on your own terms, \
before that happens.

Across episodes: the task and environment layout stay the same, but the environment \
resets. Carry forward what you learned. A new episode is marked "[Episode N]" in the \
observation.

Respond with brief reasoning if useful, then end your turn with exactly one \
\\boxed{{...}}.
"""

SYSTEM_PROMPT_STATIC = """You are an agent solving a text-based environment across multiple episodes of the \
same task. The environment and task stay the same across episodes; only the state \
resets. Use what you learned in earlier episodes to do better in later ones.

Each turn, choose one admissible action and output it as \\boxed{action text}, \
copying one of the admissible actions shown to you exactly. Respond with brief \
reasoning if useful, then end your turn with exactly one \\boxed{...}.
"""

HEURISTIC_SUMMARY_INSTRUCTION = f"""[Context management] Your interaction history has exceeded the length budget.
Summarise everything useful from the raw history below into a single paragraph
(target {SUMMARY_TOKEN_CAP} tokens): what you have learned about the environment,
what you have tried, and what remains to be done. Do not include reasoning, only
the summary itself.

History to summarise:
{{history_text}}
"""
