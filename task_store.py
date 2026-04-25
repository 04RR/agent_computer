"""Task management for deep work mode.

Provides a Task dataclass and TaskStore for creating, tracking,
and persisting tasks across agent iterations. Each session gets
its own TaskStore backed by a JSON file.
"""

from __future__ import annotations
import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("agent_computer.task_store")

# Templates of the form {{task_42.output.field}} or {{task_42.result}}.
# Used to discover cross-node references inside a task's inputs/config so the
# validator can confirm a corresponding dependency edge exists.
_TEMPLATE_PATTERN = re.compile(r"\{\{task_(\d+)\.[a-zA-Z0-9_.]+\}\}")

# Anything inside double-braces. Used to detect malformed templates the
# canonical pattern would miss (e.g. "{{4.output}}", "{{task_5}}", "{{output.task_4}}").
_ANY_TEMPLATE_PATTERN = re.compile(r"\{\{([^}]+)\}\}")


@dataclass
class Task:
    id: int
    title: str
    description: str = ""
    status: str = "pending"  # pending, in_progress, completed, blocked
    parent_id: int | None = None
    result: str = ""
    created_at: float = field(default_factory=time.time)
    # ── DAG fields (Week 1) ──
    # NOTE: depends_on encodes execution ordering and is currently NOT honored
    # by the executor — the agent loop walks tasks in id order regardless.
    # The DAG-aware scheduler lands in Week 2; until then dependencies are
    # persisted and validated but not enforced.
    node_type: str = "agent"  # "agent" | "tool" | "gather"
    depends_on: list[int] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)
    position: dict | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class CompleteResult:
    """Outcome of a TaskStore.complete() call."""
    task: Task
    success: bool
    blocked_by: list[int] = field(default_factory=list)


class TaskStore:
    """In-memory task store with JSON file persistence.

    Each mutation auto-saves to disk so state survives restarts.
    Caches sorted list, status counts, and summary text; invalidated on mutation.
    Supports deferred saves via _auto_save flag for batch operations.
    """

    VALID_STATUSES = {"pending", "in_progress", "completed", "blocked"}
    VALID_NODE_TYPES = {"agent", "tool", "gather"}

    def __init__(self, persistence_path: str | Path):
        self._path = Path(persistence_path)
        self._tasks: dict[int, Task] = {}
        self._next_id: int = 1
        # Caches
        self._sorted_cache: list[Task] | None = None
        self._summary_cache: str | None = None
        self._status_counts: dict[str, int] = {s: 0 for s in self.VALID_STATUSES}
        # Deferred save support
        self._dirty: bool = False
        self._auto_save: bool = True
        self._load()

    def _load(self):
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for td in data.get("tasks", []):
                task = Task(**td)
                self._tasks[task.id] = task
            self._next_id = data.get("next_id", 1)
            # Rebuild status counts from loaded tasks
            self._status_counts = {s: 0 for s in self.VALID_STATUSES}
            for t in self._tasks.values():
                self._status_counts[t.status] = self._status_counts.get(t.status, 0) + 1
            logger.debug(f"Loaded {len(self._tasks)} tasks from {self._path}")
        except Exception as e:
            logger.error(f"Failed to load task store: {e}")

    def _invalidate_caches(self):
        self._sorted_cache = None
        self._summary_cache = None

    def _maybe_save(self):
        if self._auto_save:
            self._save()
        else:
            self._dirty = True

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "next_id": self._next_id,
            "tasks": [t.to_dict() for t in self._tasks.values()],
        }
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._dirty = False

    def flush(self):
        """Write pending changes to disk if dirty."""
        if self._dirty:
            self._save()

    def create(
        self,
        title: str,
        description: str = "",
        parent_id: int | None = None,
        node_type: str = "agent",
        depends_on: list[int] | None = None,
        config: dict | None = None,
        inputs: dict | None = None,
        output_schema: dict | None = None,
    ):
        task = Task(
            id=self._next_id,
            title=title,
            description=description,
            parent_id=parent_id,
            node_type=node_type,
            depends_on=list(depends_on) if depends_on else [],
            config=dict(config) if config else {},
            inputs=dict(inputs) if inputs else {},
            output_schema=dict(output_schema) if output_schema else {},
        )
        self._tasks[task.id] = task
        self._next_id += 1
        self._status_counts[task.status] = self._status_counts.get(task.status, 0) + 1
        self._invalidate_caches()
        self._maybe_save()
        logger.info(f"Task created: [{task.id}] {task.title}")
        return task

    def update(self, task_id: int, **kwargs) -> Task | None:
        task = self._tasks.get(task_id)
        if not task:
            return None
        old_status = task.status
        for key, value in kwargs.items():
            if key == "status" and value not in self.VALID_STATUSES:
                continue
            if hasattr(task, key) and key not in ("id", "created_at"):
                setattr(task, key, value)
        # Update status counts if status changed
        if task.status != old_status:
            self._status_counts[old_status] = max(0, self._status_counts.get(old_status, 0) - 1)
            self._status_counts[task.status] = self._status_counts.get(task.status, 0) + 1
        self._invalidate_caches()
        self._maybe_save()
        return task

    def complete(self, task_id: int, result: str = "") -> CompleteResult | None:
        task = self._tasks.get(task_id)
        if not task:
            return None

        # Guard: don't complete a parent task if it has incomplete children
        children = [t for t in self._tasks.values() if t.parent_id == task_id]
        incomplete = [c for c in children if c.status not in ("completed",)]
        if incomplete:
            titles = ", ".join(f"[{c.id}] {c.title}" for c in incomplete[:5])
            logger.warning(
                f"Cannot complete task {task_id}: {len(incomplete)} incomplete "
                f"subtask(s): {titles}"
            )
            return CompleteResult(task=task, success=False, blocked_by=[c.id for c in incomplete])

        self.update(task_id, status="completed", result=result)
        return CompleteResult(task=task, success=True)

    def delete(self, task_id: int) -> bool:
        task = self._tasks.get(task_id)
        if task:
            self._status_counts[task.status] = max(0, self._status_counts.get(task.status, 0) - 1)
            del self._tasks[task_id]
            self._invalidate_caches()
            self._maybe_save()
            return True
        return False

    def get(self, task_id: int) -> Task | None:
        return self._tasks.get(task_id)

    def list_all(self) -> list[Task]:
        if self._sorted_cache is None:
            self._sorted_cache = sorted(self._tasks.values(), key=lambda t: t.id)
        return self._sorted_cache

    def pending_count(self) -> int:
        return self._status_counts.get("pending", 0) + self._status_counts.get("in_progress", 0)

    def completed_list(self) -> list[Task]:
        return [t for t in self.list_all() if t.status == "completed"]

    def summary(self) -> str:
        """Compact text summary for system prompt injection. Cached until mutation."""
        if self._summary_cache is not None:
            return self._summary_cache

        tasks = self.list_all()
        if not tasks:
            self._summary_cache = ""
            return ""

        parts = []
        for status in ("pending", "in_progress", "completed", "blocked"):
            c = self._status_counts.get(status, 0)
            if c > 0:
                parts.append(f"{c} {status}")

        lines = [f"Tasks: {', '.join(parts)} ({len(tasks)} total)"]

        # Build tree: top-level first, then children
        top_level = [t for t in tasks if t.parent_id is None]
        children_map: dict[int, list[Task]] = {}
        for t in tasks:
            if t.parent_id is not None:
                children_map.setdefault(t.parent_id, []).append(t)

        for t in top_level:
            lines.append(f"  [{t.id}] [{t.status}] {t.title}")
            for child in children_map.get(t.id, []):
                lines.append(f"    [{t.id}.{child.id}] [{child.status}] {child.title}")

        self._summary_cache = "\n".join(lines)
        return self._summary_cache

    def to_dict(self) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self.list_all()]

    def clear(self) -> None:
        self._tasks.clear()
        self._next_id = 1
        self._status_counts = {s: 0 for s in self.VALID_STATUSES}
        self._invalidate_caches()
        if self._path.exists():
            self._path.unlink()

    def delete_file(self) -> None:
        if self._path.exists():
            self._path.unlink()

    # ── DAG operations (Week 1: data model + validation only) ──

    def add_dependency(self, from_task: int, to_task: int) -> dict:
        """Add edge: ``to_task`` now depends on ``from_task``.

        Returns a structured result so callers can give specific feedback:
            {"ok": True}                              — edge added
            {"ok": True, "noop": True}                — edge already existed
            {"ok": False, "reason": "self_loop", "task_id": N}
            {"ok": False, "reason": "missing_task", "task_id": N}
            {"ok": False, "reason": "cycle", "would_form": [n1, n2, ...]}

        On any "ok": False return, no state is modified.
        """
        if from_task == to_task:
            return {"ok": False, "reason": "self_loop", "task_id": from_task}
        if from_task not in self._tasks:
            return {"ok": False, "reason": "missing_task", "task_id": from_task}
        if to_task not in self._tasks:
            return {"ok": False, "reason": "missing_task", "task_id": to_task}
        if from_task in self._tasks[to_task].depends_on:
            return {"ok": True, "noop": True}

        # The new edge from_task→to_task would create a cycle iff there's
        # already a depends_on path from from_task back to to_task — i.e.
        # from_task already transitively depends on to_task.
        path = self._path_via_depends(from_task, to_task)
        if path is not None:
            # path goes from_task → ... → to_task via depends_on; the new
            # edge would close the loop back to from_task.
            return {"ok": False, "reason": "cycle", "would_form": path + [from_task]}

        self._tasks[to_task].depends_on.append(from_task)
        self._invalidate_caches()
        self._maybe_save()
        return {"ok": True}

    def remove_dependency(self, from_task: int, to_task: int) -> dict:
        """Remove edge ``from_task → to_task``.

        Returns:
            {"ok": True}                              — edge removed
            {"ok": False, "reason": "no_such_edge"}   — edge didn't exist
        """
        target = self._tasks.get(to_task)
        if target is None or from_task not in target.depends_on:
            return {"ok": False, "reason": "no_such_edge"}
        target.depends_on.remove(from_task)
        self._invalidate_caches()
        self._maybe_save()
        return {"ok": True}

    def _transitive_depends(self, task_id: int) -> set[int]:
        """Set of task IDs that ``task_id`` transitively depends on.

        Excludes ``task_id`` itself. Skips IDs that no longer exist.
        """
        result: set[int] = set()
        start = self._tasks.get(task_id)
        if start is None:
            return result
        queue: deque[int] = deque(start.depends_on)
        while queue:
            tid = queue.popleft()
            if tid in result or tid == task_id:
                continue
            t = self._tasks.get(tid)
            if t is None:
                continue
            result.add(tid)
            queue.extend(t.depends_on)
        return result

    def _path_via_depends(self, start: int, target: int) -> list[int] | None:
        """BFS over depends_on from ``start`` searching for ``target``.

        Returns the path ``[start, ..., target]`` if found, else None.
        """
        if start == target:
            return [start]
        if start not in self._tasks:
            return None
        visited = {start}
        parent: dict[int, int] = {}
        queue: deque[int] = deque([start])
        while queue:
            n = queue.popleft()
            t = self._tasks.get(n)
            if t is None:
                continue
            for dep in t.depends_on:
                if dep in visited:
                    continue
                visited.add(dep)
                parent[dep] = n
                if dep == target:
                    path = [target]
                    cur = target
                    while cur in parent:
                        cur = parent[cur]
                        path.append(cur)
                    path.reverse()
                    return path
                queue.append(dep)
        return None

    def validate_dag(self) -> dict:
        """Validate DAG structure, node configuration, and template references.

        Result schema:
            {
                "valid": bool,           # False if any errors present
                "errors":   [ {...}, ... ],
                "warnings": [ {...}, ... ],
            }

        Error types: cycle, missing_dependency, unknown_node_type,
            missing_config, invalid_output_schema, templated_tool_name,
            unresolvable_template, malformed_template.
        Warning types: disconnected_components.
        """
        errors: list[dict] = []
        warnings: list[dict] = []

        tasks = self.list_all()
        tasks_by_id = {t.id: t for t in tasks}

        # 1. Per-node validation
        for t in tasks:
            if t.node_type not in self.VALID_NODE_TYPES:
                errors.append({
                    "type": "unknown_node_type",
                    "task_id": t.id,
                    "node_type": t.node_type,
                })

            for dep in t.depends_on:
                if dep not in tasks_by_id:
                    errors.append({
                        "type": "missing_dependency",
                        "task_id": t.id,
                        "references": dep,
                    })

            if t.node_type == "tool":
                tool_name = t.config.get("tool_name") if isinstance(t.config, dict) else None
                if not tool_name:
                    errors.append({
                        "type": "missing_config",
                        "task_id": t.id,
                        "field": "tool_name",
                    })
                elif isinstance(tool_name, str) and _TEMPLATE_PATTERN.search(tool_name):
                    errors.append({
                        "type": "templated_tool_name",
                        "task_id": t.id,
                        "message": (
                            "Tool node's tool_name cannot be templated — "
                            "must be known at plan time"
                        ),
                    })

            if t.node_type == "agent" and t.output_schema:
                if not isinstance(t.output_schema, dict) or "type" not in t.output_schema:
                    errors.append({
                        "type": "invalid_output_schema",
                        "task_id": t.id,
                        "message": "output_schema must be a dict containing a 'type' key",
                    })

        # 2. Template references — scan inputs AND config (stringify-and-scan).
        # A template in task M's inputs/config referencing task N is valid iff
        # M == N or N is in M's transitive depends_on. Malformed templates that
        # don't match the canonical {{task_N.field}} form are flagged separately
        # so a near-miss like {{4.output}} doesn't slip through unnoticed.
        for t in tasks:
            blob_parts: list[str] = []
            for blob in (t.inputs, t.config):
                try:
                    blob_parts.append(json.dumps(blob))
                except (TypeError, ValueError):
                    pass
            blob = "\n".join(blob_parts)
            if not blob:
                continue

            # Malformed-template scan runs unconditionally so a blob containing
            # ONLY malformed templates (no canonical refs) still gets flagged.
            canonical_inners = {
                m.group(0)[2:-2]  # strip {{ and }}
                for m in _TEMPLATE_PATTERN.finditer(blob)
            }
            all_inners = set(_ANY_TEMPLATE_PATTERN.findall(blob))
            for bad in sorted(all_inners - canonical_inners):
                errors.append({
                    "type": "malformed_template",
                    "task_id": t.id,
                    "template": "{{" + bad + "}}",
                    "message": (
                        f"Template '{{{{ {bad} }}}}' is not in the canonical "
                        f"`{{{{task_N.field}}}}` form. Use the full `task_` "
                        f"prefix and a digit ID."
                    ),
                })

            referenced = {int(m) for m in _TEMPLATE_PATTERN.findall(blob)}
            if not referenced:
                continue
            allowed = self._transitive_depends(t.id) | {t.id}
            for ref in referenced:
                if ref not in tasks_by_id:
                    errors.append({
                        "type": "unresolvable_template",
                        "task_id": t.id,
                        "template": f"task_{ref}",
                        "reason": f"task {ref} does not exist",
                    })
                elif ref not in allowed:
                    errors.append({
                        "type": "unresolvable_template",
                        "task_id": t.id,
                        "template": f"task_{ref}",
                        "reason": (
                            f"task {ref} is not in this node's transitive depends_on; "
                            f"add a dependency edge before referencing its output"
                        ),
                    })

        # 3. Cycle detection — Kahn's algorithm.
        # Leftover nodes (those never reduced to in-degree 0) are part of, or
        # downstream of, a cycle.
        if tasks:
            in_degree = {t.id: 0 for t in tasks}
            reverse_adj: dict[int, list[int]] = {t.id: [] for t in tasks}
            for t in tasks:
                for dep in t.depends_on:
                    if dep in tasks_by_id:
                        in_degree[t.id] += 1
                        reverse_adj[dep].append(t.id)
            queue: deque[int] = deque(tid for tid, deg in in_degree.items() if deg == 0)
            visited: set[int] = set()
            while queue:
                n = queue.popleft()
                visited.add(n)
                for child in reverse_adj.get(n, []):
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
            if len(visited) < len(tasks):
                cycle_ids = sorted(tid for tid in tasks_by_id if tid not in visited)
                errors.append({"type": "cycle", "task_ids": cycle_ids})

        # 4. Disconnected components (warning) — union-find on the undirected
        # version of the depends_on edge set. More than one component on a
        # multi-task plan almost always means the planner forgot to wire two
        # subgraphs together.
        if len(tasks) > 1:
            components = self._undirected_components(tasks)
            if len(components) > 1:
                warnings.append({
                    "type": "disconnected_components",
                    "components": [sorted(c) for c in components],
                    "message": (
                        f"Plan contains {len(components)} disconnected subgraphs — "
                        f"likely a planning error"
                    ),
                })

        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
        }

    def _undirected_components(self, tasks: list[Task]) -> list[set[int]]:
        """Union-find over depends_on edges treated as undirected."""
        parent = {t.id: t.id for t in tasks}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for t in tasks:
            for dep in t.depends_on:
                if dep in parent:
                    union(t.id, dep)

        groups: dict[int, set[int]] = {}
        for tid in parent:
            groups.setdefault(find(tid), set()).add(tid)
        return list(groups.values())
