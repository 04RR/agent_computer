"""Task management for deep work mode.

Provides a Task dataclass and TaskStore for creating, tracking,
and persisting tasks across agent iterations. Each session gets
its own TaskStore backed by a JSON file.
"""

from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("agent_computer.task_store")


@dataclass
class Task:
    id: int
    title: str
    description: str = ""
    status: str = "pending"  # pending, in_progress, completed, blocked
    parent_id: int | None = None
    result: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)


class TaskStore:
    """In-memory task store with JSON file persistence.

    Each mutation auto-saves to disk so state survives restarts.
    """

    VALID_STATUSES = {"pending", "in_progress", "completed", "blocked"}

    def __init__(self, persistence_path: str | Path):
        self._path = Path(persistence_path)
        self._tasks: dict[int, Task] = {}
        self._next_id: int = 1
        self._load()

    def _load(self):
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for td in data.get("tasks", []):
                task = Task(**td)
                self._tasks[task.id] = task
            self._next_id = data.get("next_id", 1)
            logger.debug(f"Loaded {len(self._tasks)} tasks from {self._path}")
        except Exception as e:
            logger.error(f"Failed to load task store: {e}")

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "next_id": self._next_id,
            "tasks": [t.to_dict() for t in self._tasks.values()],
        }
        self._path.write_text(json.dumps(data, indent=2))

    def create(self, title: str, description: str = "", parent_id: int | None = None):
        task = Task(
            id=self._next_id,
            title=title,
            description=description,
            parent_id=parent_id,
        )
        self._tasks[task.id] = task
        self._next_id += 1
        self._save()
        logger.info(f"Task created: [{task.id}] {task.title}")
        return task

    def update(self, task_id: int, **kwargs) -> Task | None:
        task = self._tasks.get(task_id)
        if not task:
            return None
        for key, value in kwargs.items():
            if key == "status" and value not in self.VALID_STATUSES:
                continue
            if hasattr(task, key) and key not in ("id", "created_at"):
                setattr(task, key, value)
        self._save()
        return task

    def complete(self, task_id: int, result: str = "") -> Task | None:
        # Guard: don't complete a parent task if it has incomplete children
        children = [t for t in self._tasks.values() if t.parent_id == task_id]
        incomplete = [c for c in children if c.status not in ("completed",)]
        if incomplete:
            titles = ", ".join(f"[{c.id}] {c.title}" for c in incomplete[:5])
            logger.warning(
                f"Cannot complete task {task_id}: {len(incomplete)} incomplete "
                f"subtask(s): {titles}"
            )
            # Return the task unchanged so the caller gets feedback
            task = self._tasks.get(task_id)
            if task:
                task._completion_blocked = True
                task._incomplete_children = [c.id for c in incomplete]
            return task
        return self.update(task_id, status="completed", result=result)

    def delete(self, task_id: int) -> bool:
        if task_id in self._tasks:
            del self._tasks[task_id]
            self._save()
            return True
        return False

    def get(self, task_id: int) -> Task | None:
        return self._tasks.get(task_id)

    def list_all(self) -> list[Task]:
        return sorted(self._tasks.values(), key=lambda t: t.id)

    def summary(self) -> str:
        """Compact text summary for system prompt injection."""
        tasks = self.list_all()
        if not tasks:
            return ""

        counts = {"pending": 0, "in_progress": 0, "completed": 0, "blocked": 0}
        for t in tasks:
            counts[t.status] = counts.get(t.status, 0) + 1

        parts = []
        for status in ("pending", "in_progress", "completed", "blocked"):
            if counts[status] > 0:
                parts.append(f"{counts[status]} {status}")

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

        return "\n".join(lines)

    def to_dict(self) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self.list_all()]

    def clear(self) -> None:
        self._tasks.clear()
        self._next_id = 1
        if self._path.exists():
            self._path.unlink()

    def delete_file(self) -> None:
        if self._path.exists():
            self._path.unlink()
