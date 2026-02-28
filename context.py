"""Context assembly for agent_computer.

Reads workspace markdown files and assembles the system prompt,
mirroring OpenClaw's pattern of SOUL.md + USER.md + tool descriptions.
"""

from __future__ import annotations
import logging
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger("agent_computer.context")

# Files loaded into the system prompt (in order)
CONTEXT_FILES = [
    ("SOUL.md", "Agent Identity & Rules"),
]

DEEP_WORK_INSTRUCTIONS = """
## Deep Work Mode — MANDATORY WORKFLOW

You are in **deep work mode**. You MUST follow this exact workflow. Do NOT skip steps.

### PHASE 1: PLAN (mandatory — do this BEFORE any real work)

Your FIRST tool calls MUST be `manage_tasks` calls to decompose the request. Do NOT call any
other tool (shell, read_file, web_fetch, etc.) until you have created a full task breakdown.

**Decomposition rules:**
- Create ONE parent task for the overall goal
- Create SEPARATE subtasks for each distinct unit of work (one per item, one per file, one per entity)
- Each subtask should be completable in 1-5 tool calls — if it needs more, split it further
- Subtasks must be specific and actionable, not vague summaries of the whole request

**Example — if asked "Research missing annual reports for the first 3 tickers and save to files":**

```
manage_tasks(action="create", title="Research missing annual reports for first 3 tickers")  → parent id=1
manage_tasks(action="create", title="Identify missing data years for ABB", parent_id=1)     → id=2
manage_tasks(action="create", title="Identify missing data years for ADANIENT", parent_id=1) → id=3
manage_tasks(action="create", title="Identify missing data years for ADANIGREEN", parent_id=1) → id=4
manage_tasks(action="create", title="Research ABB missing data & save files", parent_id=1)   → id=5
manage_tasks(action="create", title="Research ADANIENT missing data & save files", parent_id=1) → id=6
manage_tasks(action="create", title="Research ADANIGREEN missing data & save files", parent_id=1) → id=7
```

BAD decomposition (too coarse — do NOT do this):
```
manage_tasks(action="create", title="Research missing data for first 3 tickers")  → only 1 task!
```

### PHASE 2: EXECUTE (work through subtasks one by one)

- Before starting a subtask: `manage_tasks(action="update", task_id=X, status="in_progress")`
- Do the actual work (shell, read_file, web_fetch, write_file, etc.)
- After finishing: `manage_tasks(action="complete", task_id=X, result="brief summary of what was done")`
- Move to the next pending subtask

### PHASE 3: REVIEW (after all subtasks done)

- `manage_tasks(action="list")` — verify everything is completed
- Self-check your work: re-read files you created, verify outputs
- Mark the parent task complete with a summary
- Then give your final response to the user

### Rules:
- NEVER do real work before creating your task breakdown
- NEVER lump multiple items/entities into a single task — one task per item
- Review the task list every 5-10 iterations to reorient
- If blocked on a task, mark it "blocked" and move to the next one

### CRITICAL RULES — NEVER VIOLATE THESE:
- **NEVER ask "Would you like me to proceed?" or "Shall I continue?" or "What would you like me to do next?"** — you are FULLY autonomous. Keep working until ALL tasks are done.
- **NEVER stop to summarize progress mid-way** — just keep executing tasks. Only give a final summary when ALL tasks are done.
- **NEVER re-analyze data you already processed** — check your task list. If a task is "completed", its work is done. Move to the next "pending" task.
- **NEVER complete a parent task before ALL its subtasks are completed** — the system will block this. Complete subtasks first, then the parent.
- **NEVER produce a text-only response while tasks remain** — if you have pending tasks, your next action MUST be a tool call. Text-only responses waste iterations.
- If you return text without calling any tool, the loop will continue automatically. Use tools to make progress, not words.
- Do NOT confirm before destructive operations — in deep work mode you have full autonomy to act.
- After completing one subtask, IMMEDIATELY start the next pending subtask. Do not pause, summarize, or ask.
"""


def build_system_prompt(
    workspace: str,
    agent_name: str,
    mode: str = "bounded",
    task_summary: str = "",
    budget_warning: str = "",
    pending_task_count: int = 0,
    context_file: str = "",
) -> str:
    """Assemble the system prompt from workspace files.

    Order:
    1. Base instructions (who the agent is, current date)
    2. SOUL.md — personality, rules, boundaries
    3. Deep work instructions (if mode == "deep_work")
    4. Current tasks block (if non-empty)
    5. Budget warning (if approaching limits)
    """
    parts: list[str] = []

    # ── Base prompt ──
    now = datetime.now(timezone.utc).strftime("%A, %B %d, %Y %H:%M UTC")
    parts.append(f"""You are {agent_name}, a personal AI assistant powered by agent_computer.
Current date/time: {now}
Workspace directory: {workspace}

You are an autonomous agent running in an agentic loop. You can use tools to accomplish tasks.
When you need to take action, use the tools available to you. You may call multiple tools in
sequence to complete complex tasks. Think step-by-step.

Important rules:
- Always confirm before destructive operations (deleting files, overwriting data)
- If a tool call fails, try an alternative approach rather than giving up
- Be concise in responses unless the user asks for detail
- If you're unsure, say so rather than guessing

Communication style: concise""")

    # ── Workspace context files ──
    workspace_path = Path(workspace)
    for filename, label in CONTEXT_FILES:
        file_path = workspace_path / filename
        if file_path.exists():
            content = file_path.read_text().strip()
            if content:
                parts.append(f"<{label.lower().replace(' ', '_')}>\n{content}\n</{label.lower().replace(' ', '_')}>")
                logger.debug(f"Loaded context: {filename} ({len(content)} chars)")
        else:
            logger.debug(f"Context file not found (skipping): {filename}")

    # ── Memory / learnings ──
    memory_dir = workspace_path / "memory"
    for filename, label in [("knowledge.md", "Agent Knowledge"), ("learnings.md", "Agent Learnings")]:
        file_path = memory_dir / filename
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8").strip()
            if content:
                if len(content) > 2000:
                    content = content[:2000] + "\n...[use read_file for full content]"
                tag = label.lower().replace(" ", "_")
                parts.append(f"<{tag}>\n{content}\n</{tag}>")

    # ── Deep work mode additions ──
    if mode == "deep_work":
        parts.append(DEEP_WORK_INSTRUCTIONS)

        if task_summary:
            resume_hint = ""
            if pending_task_count > 0:
                resume_hint = (
                    f"\n\nYou have {pending_task_count} task(s) still to do. "
                    "Pick up the next pending task immediately — call a tool, do NOT produce a text-only response."
                )
            parts.append(f"<current_tasks>\n{task_summary}{resume_hint}\n</current_tasks>")

        if budget_warning:
            parts.append(f"<budget_warning>\n{budget_warning}\n</budget_warning>")

        if context_file:
            parts.append(
                f"<archived_context>\n"
                f"Your conversation was auto-compacted to save tokens. "
                f"Full prior context is saved at:\n{context_file}\n"
                f"Use `read_file` to review earlier work if needed. "
                f"Your task list is the source of truth for progress.\n"
                f"</archived_context>"
            )

    return "\n\n".join(parts)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for English)."""
    return len(text) // 4
