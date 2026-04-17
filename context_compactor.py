"""In-place compaction of agent conversation history.

Truncates the `content` field of large `tool_result` blocks in older messages,
leaving the message structure (roles, tool_use_ids, positions) intact. This is
called every iteration as a lightweight pre-step before the LLM call, to prevent
bloated tool results from consuming context unnecessarily.

Complementary to Session.compact() — that one archives the full conversation
to a markdown file for long-running sessions. This one is cheap GC for tool
results.
"""

from __future__ import annotations
import copy
import logging

logger = logging.getLogger("agent_computer.compactor")

_TRUNCATION_MARKER = "\n...[truncated: {removed} chars removed to save context]"


def truncate_tool_results(
    messages: list[dict],
    *,
    max_tool_result_chars: int = 2000,
    keep_recent_turns: int = 3,
) -> list[dict]:
    """Return a copy of messages with large tool_result contents truncated.

    The most recent `keep_recent_turns` turns are never touched — the agent
    likely still needs their full content for the next reasoning step.

    A "turn" for this purpose is a single message. We use a message count
    rather than a semantic turn grouping because this function doesn't
    need to preserve tool_use/tool_result pairing — truncating the content
    of a tool_result doesn't break the pairing, only removing the message
    entirely would. We never remove messages here.

    Args:
        messages: OpenAI-format message list (role/content/tool_call_id).
        max_tool_result_chars: Maximum character length to keep per tool_result
            content in the compactable region. Older content beyond this is
            replaced with a truncation marker.
        keep_recent_turns: Number of most-recent messages to leave untouched.

    Returns:
        A new list (deep-copied). The original list is not mutated.
    """
    if not messages:
        return messages

    msgs = copy.deepcopy(messages)
    protected_start = max(0, len(msgs) - keep_recent_turns)

    truncated_count = 0
    chars_removed = 0

    for i in range(protected_start):
        msg = msgs[i]

        # Only tool-role messages have tool_result content in OpenAI format
        if msg.get("role") != "tool":
            continue

        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        if len(content) <= max_tool_result_chars:
            continue

        removed = len(content) - max_tool_result_chars
        msg["content"] = (
            content[:max_tool_result_chars]
            + _TRUNCATION_MARKER.format(removed=removed)
        )
        truncated_count += 1
        chars_removed += removed

    if truncated_count > 0:
        logger.debug(
            "Truncated %d tool_result(s), freed ~%d chars (~%d tokens)",
            truncated_count, chars_removed, chars_removed // 4,
        )

    return msgs
