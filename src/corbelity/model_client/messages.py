"""Conversation history: the shape it takes on the wire, and the validation that makes
every provider behave the same way when it is malformed.

Plain mappings rather than a dataclass: every provider SDK wants dicts, so this avoids a
conversion layer on the request path.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

type Message = Mapping[str, str]
type History = Sequence[Message]

_HISTORY_ROLES = ("user", "assistant")
_MESSAGE_KEYS = frozenset({"role", "content"})


def validate_history(history: History | None) -> tuple[Message, ...]:
    """Normalize prior turns into an immutable, provider-safe tuple.

    History must strictly alternate, start with a `user` turn, and end with an
    `assistant` turn -- that last rule is what guarantees `history + [current user
    turn]` is still alternating. The providers DISAGREE about malformed lists
    (Anthropic hard-rejects, OpenRouter mostly tolerates), so validating here is what
    makes the behaviour identical everywhere instead of provider-dependent.

    Nothing is auto-repaired: silently merging same-role turns would rewrite the user's
    context without saying so. Every message names the offending index so a UI can point
    at the card. Raises ValueError, which web layers already map to a 400."""
    if not history:
        return ()

    entries = tuple(history)
    expected = "user"                      # a conversation always opens with the user
    for i, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"History entry {i} is not a message object: {entry!r}")

        missing = _MESSAGE_KEYS - set(entry)
        if missing:
            raise ValueError(f"History entry {i} is missing {', '.join(sorted(missing))}.")
        unexpected = set(entry) - _MESSAGE_KEYS
        if unexpected:
            raise ValueError(
                f"History entry {i} has unexpected key(s): {', '.join(sorted(unexpected))}."
                " Only 'role' and 'content' are allowed."
            )

        role, content = entry["role"], entry["content"]
        if not isinstance(content, str):
            raise ValueError(
                f"History entry {i} has non-string content: {type(content).__name__}."
            )
        if role not in _HISTORY_ROLES:
            raise ValueError(
                f"History entry {i} has role {role!r}; only 'user' and 'assistant' are"
                " allowed (the system prompt is set separately)."
            )
        if role != expected:
            raise ValueError(
                f"History entry {i} is a {role!r} turn where {expected!r} was expected"
                " -- turns must alternate."
            )
        expected = "assistant" if role == "user" else "user"

    if entries[-1]["role"] != "assistant":
        last = len(entries) - 1
        raise ValueError(
            f"History ends with a 'user' turn at index {last}; delete it or add the"
            " assistant reply before sending, or the prompt would follow it as a"
            " second consecutive user turn."
        )
    return entries
