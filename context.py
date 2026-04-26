"""Context assembly for agent_computer.

build_system_prompt is a pure function — all file I/O happens in the caller.
load_static_context reads workspace files once per run for caching.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger("agent_computer.context")


@dataclass
class PromptContext:
    """All inputs needed to build the agent's system prompt."""
    # Identity & setup (set once per run)
    workspace: str
    agent_name: str
    mode: str = "bounded"
    tool_names: list[str] = field(default_factory=list)
    soul_content: str = ""
    user_content: str = ""
    static_memory_fallback: str = ""
    max_iterations: int | None = None
    provider: str | None = None
    relevant_memories: list[dict] | None = None

    # Deep-work state
    deep_work_phase: str | None = None
    session_summary: str = ""

    # Dynamic (changes per iteration in deep work)
    task_summary: str = ""
    budget_warning: str = ""
    pending_task_count: int = 0
    context_file: str = ""

    # Accepted but currently unused
    user_message: str = ""


DEEP_WORK_PLANNING_INSTRUCTIONS = """
## Deep Work Mode — PLANNING PHASE

You are in the PLANNING phase of deep work mode. Your ONLY goal right now is to produce a
plan the user can review and approve. You are NOT executing the task yet.

During planning, you MUST:
- Understand what the user is asking for
- Optionally do light research (web_search, read_file, list_directory, memory_search) to
  inform your plan — only the tools listed below are available to you in this phase
- Use `manage_tasks` to create a parent task and sub-tasks that decompose the work
  * One parent task for the overall goal
  * Separate sub-tasks for each distinct unit of work (each completable in 1-5 tool calls)
  * Sub-tasks must be specific and actionable
- Write a final summary message describing: **Goal**, **Strategy**, **Tasks**, and any
  **Risks/Notes**

During planning, you MUST NOT:
- Write, edit, or delete any files
- Run shell commands of any kind
- Fetch full web pages (use `web_search` to find sources; fetching happens during execution)
- Produce the final answer to the user's question — that happens during EXECUTION,
  after approval

Even if the task seems simple enough to complete in one step, you must still plan it. The
user has asked for a plan-first workflow. Do not skip ahead.

The Completeness rules in your identity apply to the PLAN you produce — ensure your plan
addresses all parts of the user's request, but do NOT execute any parts during planning.

### DAG Planning

You are authoring a directed acyclic graph (DAG) of tasks, not a linear
list.

Every task is a node. Nodes with no dependencies run in parallel. Nodes
with `depends_on` wait for their upstream nodes to complete before
starting.

You have three node types:

1. **agent** — Runs a full agentic loop with tool access. Use when
   reasoning, decomposition, or multi-step work is needed. This is the
   default.

2. **tool** — Single deterministic tool call, no LLM. Use when the exact
   operation is known up front (fetch this URL, read this file, run
   this shell command). Faster and cheaper than an agent node. Config
   must include `tool_name` and `tool_args`.

3. **gather** — Explicit synchronization point with no computation.
   Collects outputs from all upstream nodes into a dict keyed by task
   ID. Use when multiple parallel branches converge.

### Authoring pattern

For most research-and-synthesize requests, the pattern is:

    [source_1] ─┐
    [source_2] ─┼─→ [gather] ─→ [synthesize_agent]
    [source_3] ─┘

Each `source_N` is either a tool node (if the URL/query is known) or
an agent node (if research is needed to find the source). The `gather`
node collects their outputs. The `synthesize_agent` produces the final
answer, referencing `{{gather_task_id.output}}` in its inputs.

### When to use parallel vs sequential

Default to parallel unless there's a real dependency. If task B uses
output from task A, B depends on A. Otherwise they should run in
parallel.

Example: "Research X and Y and compare" — X and Y are parallel, the
comparison depends on both (via a gather node).

Example: "Download a file and then analyze it" — sequential, the
analysis depends on the download.

### parent_id vs depends_on

The Task model has two separate fields: `parent_id` and
`depends_on`. They are NOT the same thing.

- `parent_id` is for visual grouping in the canvas UI (a future
  feature). It has NO execution semantics. Tasks with the same
  `parent_id` are visually grouped; that's it.

- `depends_on` is for execution ordering. A task with `depends_on:
  [N]` waits for task N to complete before starting, and can
  reference task N's output via templates.

Do NOT create empty "parent" agent tasks that other tasks depend on
as a grouping mechanism. If you find yourself creating a top-level
task with no real work whose only purpose is to "oversee" the
others, delete it. The DAG itself expresses the structure.

If you want N parallel branches to converge into a single output,
use a `gather` node — not a parent agent task that the branches
depend on.

### Templated inputs

Node inputs can reference upstream outputs with
`{{task_N.output.field}}` or `{{task_N.result}}`.

Always use the full `task_N` prefix. Shortened forms like
`{{N.output}}` or `{{4.output}}` are NOT valid and will fail
validation.

Where to put templates depends on the node type:

- **Agent nodes**: put templated values in the top-level `inputs`
  field of the task, not nested inside `config`. The `config`
  field is for per-node settings; `inputs` is for wiring data
  from upstream nodes.

  Correct:

      manage_tasks(
          action="create",
          title="Synthesize the answer",
          node_type="agent",
          depends_on=[4],
          inputs={"upstream_data": "{{task_4.output}}"},
          output_schema={...},
      )

  Incorrect (template buried inside config):

      manage_tasks(
          action="create",
          ...
          config={"inputs": {"upstream_data": "{{task_4.output}}"}},
          inputs={},
      )

- **Tool nodes**: templates go in `config.tool_args` because that's
  where the tool's runtime arguments live. Example:

      manage_tasks(
          action="create",
          title="Fetch a URL discovered upstream",
          node_type="tool",
          depends_on=[3],
          config={
              "tool_name": "web_fetch",
              "tool_args": {"url": "{{task_3.output.url}}"},
          },
      )

- **Gather nodes**: take no templates. They automatically collect
  upstream outputs into a dict keyed by task ID. Downstream nodes
  reference the gather as `{{task_N.output}}` and receive the full
  dict.

Template references require a dependency edge — you cannot
template task N's output into task M unless M depends on N
(directly or transitively).

### Output schemas

For agent nodes that should produce structured output, set
`output_schema` to a JSON schema describing the required shape. The
agent will be prompted to produce matching JSON.

### Workflow

1. Decompose the user's request into nodes.
2. Create each node with `manage_tasks(action="create", ...)`, including
   node_type, depends_on, config, and inputs.
3. Call `manage_tasks(action="validate")` to check for cycles, orphans,
   and unresolvable templates.
4. Fix any validation errors by creating missing nodes, adding edges
   via `connect`, or updating config.
5. Once valid, write your final summary describing the plan.

### Rules

- Every plan must have at least one source node (no dependencies) and
  at least one sink node (no nodes depend on it).
- Use `gather` nodes when 2+ parallel branches converge. Don't create
  implicit N-way joins by having one agent node depend on many upstream
  nodes — that works, but a gather node makes the synchronization
  visible and matches the expected pattern.
- Linear plans are valid. If the request really is sequential, a chain
  of nodes each depending on the previous one is correct.
- Default to agent nodes unless you're certain a tool node is
  sufficient. Agent nodes handle ambiguity; tool nodes don't.
- Always call `validate` before writing your final summary. If
  validation fails, fix the errors and re-validate.

### Phase transition
The phase will transition from PLANNING to AWAITING_APPROVAL once you produce a text response
without further tool calls. After the user approves, you will enter the EXECUTION phase with
access to all tools (shell, write_file, web_fetch, etc.).
"""

DEEP_WORK_EXECUTION_INSTRUCTIONS = """
## Deep Work Mode — EXECUTION PHASE

The user has approved your plan. Execute it now.

### WORKFLOW:
1. **Execute tasks one by one:**
   - Before starting: `manage_tasks(action="update", task_id=X, status="in_progress")`
   - Do the actual work (shell, read_file, web_fetch, write_file, etc.)
     * Within a task, batch MULTIPLE independent tool calls in one response
       (e.g., fetch 5 URLs at once - they run in parallel via asyncio.gather)
   - After finishing: `manage_tasks(action="complete", task_id=X, result="brief summary")`
   - Move to the next pending subtask

2. **Review when all subtasks are done:**
   - `manage_tasks(action="list")` — verify everything is completed
   - Self-check your work: re-read files you created, verify outputs
   - Mark the parent task complete with a summary

### Rules:
- Review the task list every 5-10 iterations to reorient
- If blocked on a task, mark it "blocked" and move to the next one

### CRITICAL RULES — NEVER VIOLATE THESE:
- **NEVER ask "Would you like me to proceed?" or "Shall I continue?"** — you are FULLY autonomous. Keep working until ALL tasks are done.
- **NEVER stop to summarize progress mid-way** — just keep executing tasks.
- **NEVER re-analyze data you already processed** — if a task is "completed", move on.
- **NEVER complete a parent task before ALL its subtasks are completed.**
- **NEVER produce a text-only response while tasks remain** — use tools to make progress.
- After completing one subtask, IMMEDIATELY start the next. Do not pause or ask.
"""

# Legacy combined instructions (kept for backward compatibility if needed)
DEEP_WORK_INSTRUCTIONS = DEEP_WORK_PLANNING_INSTRUCTIONS


def load_static_context(workspace: str, mode: str | None = None) -> dict:
    """Read static workspace files once per run. Returns dict to unpack into build_system_prompt.

    Keys: soul_content, user_content, static_memory_fallback

    The SOUL file is mode-aware: verify mode loads ``verification_soul.md``
    when present, falling back to ``SOUL.md``. Other modes always read
    ``SOUL.md``. USER.md and the memory fallbacks are mode-independent.
    """
    workspace_path = Path(workspace)
    result = {"soul_content": "", "user_content": "", "static_memory_fallback": ""}

    # Pick the soul file based on mode. Fall back to SOUL.md if the
    # mode-specific file is missing (so verify mode still works on a fresh
    # workspace before the verification SOUL is created).
    if mode == "verify":
        soul_candidates = ("verification_soul.md", "SOUL.md")
    else:
        soul_candidates = ("SOUL.md",)

    for fname in soul_candidates:
        path = workspace_path / fname
        if path.exists():
            content = path.read_text(encoding="utf-8").strip()
            if content:
                result["soul_content"] = content
                logger.debug(f"Loaded {fname} ({len(content)} chars) for mode={mode!r}")
                break

    user_path = workspace_path / "USER.md"
    if user_path.exists():
        content = user_path.read_text(encoding="utf-8").strip()
        if content:
            result["user_content"] = content
            logger.debug(f"Loaded USER.md ({len(content)} chars)")

    # Static memory fallback (used when MemorySearch is unavailable)
    memory_dir = workspace_path / "memory"
    parts = []
    for filename, label in [("knowledge.md", "Agent Knowledge"), ("learnings.md", "Agent Learnings")]:
        file_path = memory_dir / filename
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8").strip()
            if content:
                if len(content) > 2000:
                    content = content[:2000] + "\n...[use read_file for full content]"
                tag = label.lower().replace(" ", "_")
                parts.append(f"<{tag}>\n{content}\n</{tag}>")
    result["static_memory_fallback"] = "\n\n".join(parts)

    return result


def build_static_prompt_prefix(ctx: PromptContext) -> str:
    """Assemble the static parts of the system prompt (parts 1-7).

    These don't change between iterations within a single deep work run.
    """
    parts: list[str] = []

    # 1. Base identity
    now = datetime.now(timezone.utc).strftime("%A, %B %d, %Y %H:%M UTC")
    parts.append(f"""You are {ctx.agent_name}, a personal AI assistant powered by agent_computer.
Current date/time: {now}
Workspace directory: {ctx.workspace}

You are an autonomous agent running in an agentic loop with tool access.
Think step-by-step. Be concise. If a tool call fails, try an alternative approach.

IMPORTANT: You can make MULTIPLE tool calls in a SINGLE response when operations are independent.
Examples of when to batch:
- Research: Fetch multiple URLs at once (web_fetch in parallel)
- File operations: Read several files simultaneously
- Data gathering: Query multiple sources concurrently

Batched tool calls execute in parallel for faster results. Only make sequential calls when
one operation depends on another's output.""")

    # 1.5. Efficiency rules for bounded mode on local models
    if ctx.mode != "deep_work" and ctx.provider == "lmstudio" and ctx.max_iterations is not None:
        parts.append(f"""## Efficiency Rules
- You have a HARD LIMIT of {ctx.max_iterations} total steps. Plan accordingly.
- Make MULTIPLE tool calls in a SINGLE response when possible (they run in parallel).
- For web research: fetch 2-3 URLs in ONE response, not one at a time.
- After gathering information, STOP fetching and give your answer. Do NOT keep searching for marginally better results.
- If you have enough information to answer the user, ANSWER IMMEDIATELY.""")

    # 2. SOUL.md
    if ctx.soul_content:
        parts.append(f"<agent_identity_&_rules>\n{ctx.soul_content}\n</agent_identity_&_rules>")

    # 3. USER.md
    if ctx.user_content:
        parts.append(f"<user_context>\n{ctx.user_content}\n</user_context>")

    # 4. Tool inventory
    if ctx.tool_names:
        tool_list = ", ".join(ctx.tool_names)
        parts.append(
            f"<available_tools>\nYou have these tools: {tool_list}\n"
            "Use the appropriate tool for each task. If a tool call fails, try an alternative.\n"
            "</available_tools>"
        )

    # 4.5. Parallel tool usage examples (deep work + verify both use DAGs)
    if ctx.mode in ("deep_work", "verify"):
        parts.append("""<parallel_tool_examples>
GOOD - Batched (1 iteration):
Call web_fetch("url1"), web_fetch("url2"), web_fetch("url3") in ONE response
→ Runtime executes in parallel → All results returned together

BAD - Sequential (3 iterations):
Call web_fetch("url1") → wait → response → next iteration
Call web_fetch("url2") → wait → response → next iteration
Call web_fetch("url3") → wait → response → next iteration

When NOT to batch:
- Operations depending on each other (read config → use config values)
- State modifications (write file → read it back)
</parallel_tool_examples>""")

    # 5. Relevant memories
    if ctx.relevant_memories:
        mem_lines = []
        for m in ctx.relevant_memories:
            content = m.get("content", "")
            if len(content) > 500:
                content = content[:500] + "..."
            mem_lines.append(
                f"- [{m.get('source_type', '?')}] **{m.get('title', 'Untitled')}**: {content}"
            )
        parts.append(
            "<relevant_memories>\n"
            "The following memories may be relevant:\n"
            + "\n".join(mem_lines)
            + "\n</relevant_memories>"
        )
    elif ctx.static_memory_fallback:
        parts.append(ctx.static_memory_fallback)

    # 6. Session context
    if ctx.session_summary:
        parts.append(f"<session_context>\nThis session so far: {ctx.session_summary}\n</session_context>")

    # 7. DAG-mode instructions (phase-aware). Verify mode reuses the same
    # DAG planning/execution mechanics — verification IS a fan-out/gather/
    # synthesize task. The verification SOUL provides the domain framing.
    if ctx.mode in ("deep_work", "verify"):
        if ctx.deep_work_phase == "executing":
            parts.append(DEEP_WORK_EXECUTION_INSTRUCTIONS)
        else:
            parts.append(DEEP_WORK_PLANNING_INSTRUCTIONS)

    return "\n\n".join(parts)


def build_dynamic_suffix(ctx: PromptContext) -> str:
    """Assemble the dynamic parts of the system prompt (parts 8-10).

    These change per iteration in deep work mode.
    """
    parts: list[str] = []

    # 8. Task state
    if ctx.task_summary:
        resume_hint = ""
        if ctx.pending_task_count > 0:
            resume_hint = (
                f"\n\nYou have {ctx.pending_task_count} task(s) still to do. "
                "Pick up the next pending task immediately — call a tool, do NOT produce a text-only response."
            )
        parts.append(f"<current_tasks>\n{ctx.task_summary}{resume_hint}\n</current_tasks>")

    # 9. Budget warning
    if ctx.budget_warning:
        parts.append(f"<budget_warning>\n{ctx.budget_warning}\n</budget_warning>")

    # 10. Archived context
    if ctx.context_file:
        parts.append(
            f"<archived_context>\n"
            f"Your conversation was auto-compacted to save tokens. "
            f"Full prior context is saved at:\n{ctx.context_file}\n"
            f"Use `read_file` to review earlier work if needed. "
            f"Your task list is the source of truth for progress.\n"
            f"</archived_context>"
        )

    return "\n\n".join(parts)


def build_system_prompt(ctx: PromptContext) -> str:
    """Assemble the full system prompt. Pure function — no file I/O."""
    prefix = build_static_prompt_prefix(ctx)
    suffix = build_dynamic_suffix(ctx)
    return prefix + ("\n\n" + suffix if suffix else "")


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for English)."""
    return len(text) // 4
