"""Shared TextWorld thread-safety patch.

TextWorld uses a handful of module-level ``_PARSER`` singletons (tatsu
parsers) whose ``.parse()`` method mutates internal stacks. Concurrent
``reset()``/``step()`` calls from different rollout threads corrupt those
stacks (``IndexError: pop from empty list``), which matters a lot for us
since GRPO rollouts run many envs in parallel.

ORBIT's ``envs/alfworld_env_adapter.py`` already carries an identical patch
(applied at import time), but we can't rely on ALFWorld being installed —
this project targets *raw* TextWorld, not the ALFWorld household domain.
So we duplicate the same patch here as a standalone, idempotent function and
call it from ``textworld_env_adapter.py`` directly.

Safe to call multiple times (e.g. if both this module and ALFWorld's adapter
get imported in the same process) — each call just re-installs the same
per-thread-local factories.
"""

from __future__ import annotations

import threading

_PATCH_LOCK = threading.Lock()
_PATCHED = False

# Guards TextWorld's global gym registration, which is not thread-safe.
TW_REGISTER_LOCK = threading.Lock()


def install_threadlocal_parsers() -> None:
    """Patch TextWorld's `_parse_and_convert` functions to use per-thread parsers."""
    global _PATCHED
    with _PATCH_LOCK:
        if _PATCHED:
            return

        import textworld.envs.pddl.logic as _pddl_logic
        import textworld.envs.pddl.textgen as _textgen
        import textworld.logic as _tw_logic
        from textworld.envs.pddl.logic.model import PddlLogicModelBuilderSemantics
        from textworld.envs.pddl.logic.parser import PddlLogicParser
        from textworld.envs.pddl.textgen.model import CSGModelBuilderSemantics
        from textworld.envs.pddl.textgen.parser import CSGParser
        from textworld.logic.model import GameLogicModelBuilderSemantics
        from textworld.logic.parser import GameLogicParser

        def _make_threadsafe(module, parser_factory, walker_cls):
            tls = threading.local()

            def _parse_and_convert(*args, **kwargs):
                try:
                    parser = tls.parser
                except AttributeError:
                    parser = parser_factory()
                    tls.parser = parser
                model = parser.parse(*args, **kwargs)
                return walker_cls().walk(model)

            module._parse_and_convert = _parse_and_convert

        _make_threadsafe(
            _pddl_logic,
            lambda: PddlLogicParser(semantics=PddlLogicModelBuilderSemantics(), parseinfo=True),
            _pddl_logic._ModelConverter,
        )
        _make_threadsafe(
            _textgen,
            lambda: CSGParser(semantics=CSGModelBuilderSemantics(), parseinfo=True),
            _textgen._Converter,
        )
        _make_threadsafe(
            _tw_logic,
            lambda: GameLogicParser(semantics=GameLogicModelBuilderSemantics(), parseinfo=True),
            _tw_logic._ModelConverter,
        )

        # textworld.logic also calls _PARSER.parse directly in parse_document().
        _tw_logic_tls = threading.local()

        class _TwLogicParserProxy:
            def __getattr__(self, name):
                try:
                    p = _tw_logic_tls.parser
                except AttributeError:
                    p = GameLogicParser(semantics=GameLogicModelBuilderSemantics(), parseinfo=True)
                    _tw_logic_tls.parser = p
                return getattr(p, name)

        _tw_logic._PARSER = _TwLogicParserProxy()

        _PATCHED = True
