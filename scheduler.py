"""DAG scheduler — Week 2.

Replaces the linear task walker with a real DAG executor:

  - resolves {{task_N.field.subfield}} templates against completed
    upstream task outputs
  - dispatches tool / gather / agent nodes per type
  - runs ready tasks in parallel via asyncio.gather
  - manages status transitions (pending → in_progress → completed/failed)
  - propagates failures via mark_blocked_dependents
  - hands off to a post-DAG synthesis step that produces the final
    Markdown report

This module is consumed by both verify mode and deep_work mode.
Bounded mode keeps the existing linear executor — only DAG modes
use the scheduler.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

logger = logging.getLogger("agent_computer.scheduler")


# Templates inside double braces. Captures the inner expression.
_TEMPLATE_INLINE_PATTERN = re.compile(r"\{\{([^}]+)\}\}")


class TemplateResolutionError(Exception):
    """Raised when {{task_N.field}} can't be resolved at runtime."""


class SchedulerError(Exception):
    """Raised on deadlocks or other scheduler-internal invariant failures."""


# ─── Template resolution ────────────────────────────────────────────────

def resolve_templates(value: Any, task_store) -> Any:
    """Walk a dict / list / string recursively. Substitute every
    {{task_N.field.subfield}} reference with the resolved upstream
    output. Strings without templates pass through unchanged.

    Recognized syntax inside braces:
      task_N                          — shorthand for task_N.output
      task_N.output                   — task N's full output
      task_N.output.field             — nested field
      task_N.output.field.sub.deeper  — deeper nesting
      task_N.result                   — the result string field
      task_N.status                   — the task's status string

    If the entire string is a single template, the resolved value
    is returned in its native type (dict/list/str/number/bool/None).
    If the template is embedded in a larger string, the resolved
    value is JSON-stringified for inline substitution.

    Raises TemplateResolutionError on:
      - reference to a task that doesn't exist
      - reference to a task whose status is not "completed"
      - a field path that doesn't exist on the resolved output
    """
    if isinstance(value, str):
        return _resolve_string(value, task_store)
    if isinstance(value, dict):
        return {k: resolve_templates(v, task_store) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_templates(v, task_store) for v in value]
    return value


def _resolve_string(text: str, task_store) -> Any:
    """If the entire string is one template, return native value;
    otherwise substitute templates inline and return a string."""
    matches = list(_TEMPLATE_INLINE_PATTERN.finditer(text))
    if not matches:
        return text

    # Whole-string single-template case: return the resolved value
    # in its native type so dicts/lists/numbers don't get stringified.
    if (
        len(matches) == 1
        and matches[0].start() == 0
        and matches[0].end() == len(text)
    ):
        return _resolve_one(matches[0].group(1), task_store)

    # Embedded — substitute each match with a JSON string of its value.
    out_parts: list[str] = []
    cursor = 0
    for m in matches:
        out_parts.append(text[cursor:m.start()])
        resolved = _resolve_one(m.group(1), task_store)
        if isinstance(resolved, str):
            out_parts.append(resolved)
        else:
            out_parts.append(json.dumps(resolved, ensure_ascii=False))
        cursor = m.end()
    out_parts.append(text[cursor:])
    return "".join(out_parts)


def _resolve_one(inner: str, task_store) -> Any:
    """Resolve a single template's inner expression, e.g.
    'task_5.output.task_2'. Returns the resolved value.
    """
    inner = inner.strip()
    parts = inner.split(".")
    if not parts or not parts[0].startswith("task_"):
        raise TemplateResolutionError(
            f"malformed template inner expression: {{{{{inner}}}}}"
        )
    head = parts[0]
    try:
        task_id = int(head[len("task_"):])
    except ValueError:
        raise TemplateResolutionError(
            f"malformed task id in template: {{{{{inner}}}}}"
        )

    task = task_store.get(task_id)
    if task is None:
        raise TemplateResolutionError(
            f"template references task {task_id} which does not exist"
        )
    if task.status != "completed":
        raise TemplateResolutionError(
            f"template references task {task_id} (status={task.status!r}) "
            f"which has not completed yet"
        )

    # Decide which task field is being addressed.
    if len(parts) == 1:
        # {{task_N}} — shorthand for output
        return task.output
    field = parts[1]
    rest = parts[2:]

    if field == "output":
        cursor: Any = task.output
    elif field == "result":
        cursor = task.result
    elif field == "status":
        cursor = task.status
    elif field in ("title", "description", "id"):
        cursor = getattr(task, field)
    else:
        raise TemplateResolutionError(
            f"unknown task field {field!r} in template {{{{{inner}}}}}"
        )

    # Walk the remaining path through the resolved value.
    for key in rest:
        if isinstance(cursor, dict):
            if key not in cursor:
                raise TemplateResolutionError(
                    f"template {{{{{inner}}}}}: field {key!r} not in task {task_id}'s "
                    f"resolved value (available keys: {sorted(cursor.keys())})"
                )
            cursor = cursor[key]
        elif isinstance(cursor, list):
            try:
                cursor = cursor[int(key)]
            except (ValueError, IndexError):
                raise TemplateResolutionError(
                    f"template {{{{{inner}}}}}: cannot index list with {key!r} "
                    f"in task {task_id}"
                )
        else:
            raise TemplateResolutionError(
                f"template {{{{{inner}}}}}: cannot walk into "
                f"non-dict/non-list value at {key!r} (got {type(cursor).__name__})"
            )

    return cursor


# ─── DagScheduler ───────────────────────────────────────────────────────


class DagScheduler:
    """Execute a validated DAG of tasks to completion.

    The agent loop is NOT involved during DAG execution. Tool nodes are
    dispatched via the registry (resolving templated args first). Gather
    nodes synthesize an output dict from upstream outputs. Agent nodes
    spawn a single bounded LLM call to produce structured output (no tool
    use within the agent node).

    Failures cascade: when a task fails, its transitive dependents are
    marked 'blocked' and the scheduler proceeds with whatever's still
    runnable. The synthesis step downstream sees the partial results.
    """

    def __init__(
        self,
        task_store,
        tool_registry,
        agent,
        mode: str,
        session_id: str | None = None,
    ):
        self.task_store = task_store
        self.tool_registry = tool_registry
        self.agent = agent
        self.mode = mode
        self.session_id = session_id

    async def run(self) -> dict:
        """Execute the DAG to completion. Returns a summary dict with
        completed / failed / blocked task IDs and elapsed time.

        Raises SchedulerError on a deadlock (no ready tasks but pending
        tasks remain — should be impossible if the validator caught
        cycles, but we check defensively).
        """
        t0 = time.monotonic()
        completed: list[int] = []
        failed: list[dict] = []

        while True:
            # Anything still pending?
            pending = [
                t for t in self.task_store.list_all()
                if t.status == "pending"
            ]
            if not pending:
                break

            ready = self.task_store.list_ready()
            if not ready:
                # Pending tasks exist but none are ready — either all
                # are blocked (cascade) or we have a real deadlock.
                blocked_count = sum(
                    1 for t in self.task_store.list_all() if t.status == "blocked"
                )
                pending_count = len(pending)
                if pending_count == 0:
                    break
                if blocked_count > 0 and len(ready) == 0:
                    # Genuine cascade: every remaining pending task
                    # has at least one failed/blocked upstream. Block
                    # them all and exit.
                    for t in pending:
                        self.task_store.set_status(t.id, "blocked")
                    break
                raise SchedulerError(
                    f"deadlock: {pending_count} tasks pending but none ready"
                )

            logger.info(
                f"DagScheduler dispatching {len(ready)} task(s) in parallel: "
                f"{[(t.id, t.node_type, t.title) for t in ready]}"
            )

            results = await asyncio.gather(
                *(self._dispatch(t) for t in ready),
                return_exceptions=True,
            )

            for t, res in zip(ready, results):
                if isinstance(res, Exception):
                    # Defensive — _dispatch already converts internal
                    # errors. This path catches truly unexpected ones.
                    self.task_store.set_error(t.id, f"unexpected: {res}")
                    self.task_store.mark_blocked_dependents(t.id)
                    failed.append({"id": t.id, "error": str(res)})
                    continue
                status = res.get("status")
                if status == "completed":
                    completed.append(t.id)
                elif status == "failed":
                    failed.append({"id": t.id, "error": res.get("error", "")})

        # Anything not completed at this point is failed or blocked.
        all_blocked = [
            t.id for t in self.task_store.list_all() if t.status == "blocked"
        ]
        elapsed_ms = round((time.monotonic() - t0) * 1000)

        if failed:
            overall = "partial" if completed else "failed"
        else:
            overall = "completed"

        return {
            "status": overall,
            "completed_tasks": completed,
            "failed_tasks": failed,
            "blocked_tasks": all_blocked,
            "elapsed_ms": elapsed_ms,
        }

    async def _dispatch(self, task) -> dict:
        if task.node_type == "tool":
            return await self._dispatch_tool(task)
        if task.node_type == "gather":
            return await self._dispatch_gather(task)
        if task.node_type == "agent":
            return await self._dispatch_agent(task)
        # Unknown — fail it explicitly so synthesis sees the error.
        self.task_store.set_error(task.id, f"unknown node_type: {task.node_type!r}")
        self.task_store.mark_blocked_dependents(task.id)
        return {"task_id": task.id, "status": "failed", "error": "unknown_node_type"}

    # ── Tool node dispatch ──

    async def _dispatch_tool(self, task) -> dict:
        self.task_store.set_status(task.id, "in_progress")

        tool_name = task.config.get("tool_name") if isinstance(task.config, dict) else None
        if not tool_name:
            err = "tool node missing config.tool_name"
            self.task_store.set_error(task.id, err)
            self.task_store.mark_blocked_dependents(task.id)
            return {"task_id": task.id, "status": "failed", "error": err}

        raw_args = task.config.get("tool_args", {}) if isinstance(task.config, dict) else {}
        try:
            resolved_args = resolve_templates(raw_args, self.task_store)
        except TemplateResolutionError as e:
            self.task_store.set_error(task.id, f"template: {e}")
            self.task_store.mark_blocked_dependents(task.id)
            return {"task_id": task.id, "status": "failed", "error": str(e)}

        # Build the tool context the registry forwards to handlers that
        # accept _context. Includes dag_mode_active so manage_tasks can
        # detect runtime status updates from agent nodes (Step 7).
        tool_context = {
            "task_store": self.task_store,
            "mode": self.mode,
            "session_id": self.session_id,
            "dag_mode_active": True,
        }

        try:
            result_str = await self.tool_registry.execute(
                tool_name, resolved_args, context=tool_context
            )
        except Exception as e:
            self.task_store.set_error(task.id, f"tool crash: {e}")
            self.task_store.mark_blocked_dependents(task.id)
            return {"task_id": task.id, "status": "failed", "error": str(e)}

        # Tools return JSON strings per the existing contract. Parse for
        # storage so downstream templates can resolve into the structure.
        try:
            output = json.loads(result_str) if isinstance(result_str, str) else result_str
        except json.JSONDecodeError:
            output = {"_raw": result_str}

        # If the tool returned an {"error": ...} payload, the tool ran
        # but returned an error state — that's still a "completed" tool
        # call from the scheduler's perspective (it didn't crash). The
        # error is part of the output for synthesis to surface. Tool
        # crashes (caught above) are different — those mark failed.
        self.task_store.set_output(task.id, output)
        self.task_store.set_status(task.id, "completed")
        return {"task_id": task.id, "status": "completed"}

    # ── Gather node dispatch ──

    async def _dispatch_gather(self, task) -> dict:
        """Gather nodes don't run tools or LLMs. They produce an output
        dict keyed by upstream task ID. Downstream templates access via
        {{task_<gather_id>.output.task_<dep_id>}}.
        """
        self.task_store.set_status(task.id, "in_progress")
        try:
            output: dict = {}
            for dep_id in task.depends_on:
                dep = self.task_store.get(dep_id)
                if dep is None:
                    raise SchedulerError(f"gather dep task {dep_id} missing")
                output[f"task_{dep_id}"] = dep.output
            self.task_store.set_output(task.id, output)
            self.task_store.set_status(task.id, "completed")
            return {"task_id": task.id, "status": "completed"}
        except Exception as e:
            self.task_store.set_error(task.id, f"gather: {e}")
            self.task_store.mark_blocked_dependents(task.id)
            return {"task_id": task.id, "status": "failed", "error": str(e)}

    # ── Agent node dispatch ──

    async def _dispatch_agent(self, task) -> dict:
        """Single bounded LLM call producing structured output for this node.

        Verify mode never authors agent nodes (per the SOUL update).
        Deep_work plans may author them — this is the dispatch path.

        No tool use within the call. The agent at this layer is not allowed
        to enter a tool-using loop; its job is to emit JSON matching the
        node's output_schema (or a free-form JSON object if no schema).
        """
        self.task_store.set_status(task.id, "in_progress")

        # Resolve templated inputs
        try:
            resolved_inputs = resolve_templates(task.inputs or {}, self.task_store)
        except TemplateResolutionError as e:
            self.task_store.set_error(task.id, f"template: {e}")
            self.task_store.mark_blocked_dependents(task.id)
            return {"task_id": task.id, "status": "failed", "error": str(e)}

        # Build the prompt for this single-shot agent call.
        schema = task.output_schema if isinstance(task.output_schema, dict) else {}
        schema_blob = json.dumps(schema, indent=2) if schema else "(no specific schema — produce a useful JSON object)"
        inputs_blob = json.dumps(resolved_inputs, indent=2, ensure_ascii=False)

        system_prompt = (
            "You are an agent node inside a DAG. You produce a structured "
            "JSON output for one specific subtask, then exit. You DO NOT "
            "call tools or enter a tool-using loop. Your only job is to "
            "produce valid JSON matching the requested schema.\n\n"
            f"Subtask: {task.title}\n"
            f"Description: {task.description or '(no extra description)'}\n\n"
            f"Required output schema:\n{schema_blob}\n\n"
            "Respond with valid JSON only. No markdown fences. No commentary."
        )
        user_msg = (
            f"Resolved inputs from upstream tasks:\n{inputs_blob}\n\n"
            "Produce the JSON output now."
        )

        # JSON-mode if the provider supports it (lmstudio is the holdout).
        provider = self.agent.agent_config.model.provider
        kwargs: dict = {
            "model": self.agent.agent_config.model.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 2000,
            "temperature": 0.2,
        }
        if provider not in ("lmstudio", ""):
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = await self.agent.client.chat.completions.create(**kwargs)
            text = (response.choices[0].message.content or "").strip()
            # Tolerate ```json fences in case the model defies instructions
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*\n?", "", text)
                text = re.sub(r"\n?```\s*$", "", text)
            output = json.loads(text)
        except json.JSONDecodeError as e:
            err = f"agent node returned non-JSON: {e}; raw={text[:200]!r}"
            self.task_store.set_error(task.id, err)
            self.task_store.mark_blocked_dependents(task.id)
            return {"task_id": task.id, "status": "failed", "error": err}
        except Exception as e:
            self.task_store.set_error(task.id, f"agent node LLM call: {e}")
            self.task_store.mark_blocked_dependents(task.id)
            return {"task_id": task.id, "status": "failed", "error": str(e)}

        self.task_store.set_output(task.id, output)
        self.task_store.set_status(task.id, "completed")
        return {"task_id": task.id, "status": "completed"}


# ─── Post-DAG synthesis ─────────────────────────────────────────────────

_VERIFICATION_SYNTHESIS_PROMPT = """\
You are producing the final structured Markdown report for an image
verification request. The DAG has finished; the task outputs below are
your only source of evidence.

Produce a Markdown report with these sections, in order:

### Caption claims
What the caption claimed, decomposed into who / what / when / where /
source. Pull from the extract_caption_claims task output.

### Image provenance
Where the image has appeared on the web. Use the reverse_image_search
output: first-seen date and domain, the spread of crawl dates, and
the top matched domains. Whether that pattern is consistent with the
caption's claimed time and location.

If reverse_image_search output has `_stub: true`, you MUST state in
this section that the search results are simulated stub data, not
real TinEye results. Otherwise readers will trust fixture domains as
real evidence.

### Metadata
Camera make/model, EXIF datetime, GPS, software field, and any flagged
anomalies (future-dated EXIF, AI-generator software, missing EXIF, GPS
without camera). Pull from the extract_image_metadata task output.

### Fact-check matches
What mainstream fact-checkers have said about this claim or image.
Include publisher names and ratings if any matches were returned.
"No fact-check matches" is a finding, not a gap — say so explicitly
when the count is zero.

### Reconciliation
For each dimension (when / where / who / what), one short paragraph:
the caption's claim, the evidence, the verdict (consistent /
contradicts / inconclusive), and one sentence of reasoning. You
perform the cross-check yourself directly from the four task
outputs — there is no separate reconcile tool result. "Inconclusive"
is the default whenever evidence is missing or mixed; do not promote
to "consistent" without affirmative evidence.

### Bottom line
One paragraph summarizing what the evidence suggests. NOT a binary
verdict.

Framing rules — never violate:

1. Never use the words "fake", "real", or "genuine". Acceptable
   framings: "evidence suggests misattribution", "consistent with the
   caption's date and location", "inconclusive — image has no web
   history and no metadata".
2. "No web matches" does NOT mean AI-generated. A real photo taken
   minutes ago has no web history. Be explicit about this whenever
   reverse_image_search returned no matches.
3. Hedge appropriately: "suggests", "appears consistent with",
   "is inconsistent with". Avoid bare "is" verdicts.
4. If any task in the DAG failed or is blocked, acknowledge it rather
   than papering over: "fact_check_lookup was unavailable" instead of
   silently omitting the section.

Output only the Markdown report. No tool calls. No commentary outside
the report sections.
"""


_DEEP_WORK_SYNTHESIS_PROMPT = """\
You are producing the final answer to the user's original request. The
DAG has finished; the task outputs below are your evidence base.

Synthesize a clear, direct answer that draws on those outputs. Cite
which task each fact came from when it would help the reader trust
the answer.

If any task failed or is blocked, acknowledge what's missing rather
than papering over with general knowledge — your job is to surface
what the DAG produced, not to fabricate around its gaps.

Output only the answer. No tool calls. No meta-commentary about how
the DAG was structured unless the structure itself is the answer the
user asked for.
"""


def _build_synthesis_context(task_store) -> str:
    """Build a compact context blob describing every task's outcome,
    for inclusion in the synthesis prompt."""
    lines: list[str] = []
    for t in task_store.list_all():
        header = f"Task {t.id} ({t.title}, {t.node_type}, status={t.status})"
        lines.append(header)
        if t.status == "completed":
            try:
                blob = json.dumps(t.output, indent=2, ensure_ascii=False)
            except (TypeError, ValueError):
                blob = repr(t.output)
            # Cap individual blobs to keep the synthesis prompt bounded.
            if len(blob) > 4000:
                blob = blob[:4000] + "\n...[truncated]"
            lines.append(f"  output:\n{blob}")
        elif t.status in ("failed", "blocked"):
            lines.append(f"  result: {t.result or '(no detail)'}")
        lines.append("")
    return "\n".join(lines)


async def synthesize_dag_report(
    agent,
    task_store,
    mode: str,
    original_request: str,
    max_tokens: int = 3000,
) -> str:
    """One LLM call. No tool use. Produces the final report.

    Picks the verification or deep_work prompt based on mode. The
    original_request is the user's first message (e.g. for verify
    mode the 'VERIFICATION REQUEST...' blob with image path + caption).
    """
    system_prompt = (
        _VERIFICATION_SYNTHESIS_PROMPT
        if mode == "verify"
        else _DEEP_WORK_SYNTHESIS_PROMPT
    )
    context_blob = _build_synthesis_context(task_store)
    user_msg = (
        f"Original request:\n\n{original_request}\n\n"
        f"=== DAG execution results ===\n\n{context_blob}\n\n"
        f"Produce the final report now."
    )
    try:
        response = await agent.client.chat.completions.create(
            model=agent.agent_config.model.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error(f"synthesis LLM call failed: {e}")
        return f"[Synthesis failed: {e}]"
