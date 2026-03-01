# agent_computer

A Python-based autonomous agent framework with tool use, long-term memory,
scheduled tasks, and auto-reflection. Supports local models via **LM Studio**
and cloud models via **OpenRouter**.

## Core Concepts

- **Gateway** — FastAPI process handling WebSocket + HTTP connections
- **Agent Runtime** — Agentic loop: context assembly → LLM → tool execution → reply
- **Sessions** — Stateful conversations persisted as JSONL, with serial execution
- **Tools** — Modular, registerable capabilities (shell, files, web scraping, tasks, memory search)
- **Workspace** — Identity files (SOUL.md, USER.md) and agent memory
- **Deep Work Mode** — Extended autonomous execution with task decomposition, auto-compaction, and token budgets
- **Reflection Engine** — Extracts knowledge, learnings, and reusable skills from completed sessions
- **Memory Search** — Hybrid vector + keyword search (BM25 + cosine similarity via Reciprocal Rank Fusion)
- **Cron Scheduler** — Scheduled agent tasks that run on a timer or at startup
- **Skill Loader** — Auto-discovered Python skills registered as tools at runtime

## Quick Start

```bash
pip install -r requirements.txt

# Option A: Local models via LM Studio (default)
# Make sure LM Studio is running with an API server at http://10.5.0.2:1234
python gateway.py

# Option B: Cloud models via OpenRouter
export OPENROUTER_API_KEY=sk-or-...
# Then change provider to "openrouter" in config.json
python gateway.py
```

Open `http://localhost:8000` for the web UI, or connect via WebSocket at
`ws://localhost:8000/ws/{session_id}`.

## Configuration

Edit `config.json` to configure the agent. All fields have sensible defaults.

```json
{
  "gateway": { "host": "0.0.0.0", "port": 8000 },
  "agent": {
    "name": "agent_computer Agent",
    "workspace": "./workspace",
    "model": {
      "provider": "lmstudio",
      "model_id": "your-model-id",
      "base_url": "http://10.5.0.2:1234/v1",
      "max_tokens": 8192
    },
    "tools": {
      "allow": ["shell", "read_file", "write_file", "list_directory",
                 "web_fetch", "web_fetch_js", "web_fetch_stealth",
                 "manage_tasks", "memory_search"],
      "require_approval": ["shell"]
    },
    "max_loop_iterations": 15,
    "deep_work": {
      "max_iterations": 200,
      "token_budget": 1500000,
      "warning_threshold": 0.8
    }
  },
  "sessions": { "directory": "./sessions", "max_history_tokens": 100000 },
  "reflection": {
    "enabled": true,
    "model_id": "",
    "max_tokens": 4096,
    "max_sessions_per_startup": 10
  },
  "memory": {
    "enabled": true,
    "embedding_base_url": "http://10.5.0.2:1234/v1",
    "embedding_model": "text-embedding-nomic-embed-text-v1.5",
    "top_k": 5
  }
}
```

### Switching Models

Models can be changed in `config.json` or at runtime via the API:

```bash
# Runtime model switch
curl -X POST http://localhost:8000/api/models/select \
  -H "Content-Type: application/json" \
  -d '{"provider": "openrouter", "model_id": "anthropic/claude-sonnet-4-6"}'
```

**Providers:**
- `lmstudio` — Local models via LM Studio (default)
- `openrouter` — Cloud models (Claude, GPT, Gemini, Llama, DeepSeek, etc.)

## Project Structure

```
agent_computer/
├── gateway.py            # FastAPI gateway (entry point)
├── agent.py              # Agent runtime + agentic loop
├── session.py            # Session management, JSONL persistence, compaction
├── tool_registry.py      # Tool registration and execution
├── context.py            # System prompt assembly from workspace files
├── config.py             # Configuration loader with defaults
├── config.json           # Runtime configuration
├── task_store.py         # Task management for deep work mode
├── cron.py               # Cron scheduler (scheduled agent tasks)
├── reflection.py         # Auto-reflection engine (knowledge extraction)
├── memory_search.py      # Hybrid vector + keyword memory search (SQLite)
├── skill_loader.py       # Dynamic skill loader from Python files
├── tools/
│   ├── __init__.py       # Built-in tools (shell, files, tasks, memory)
│   └── web_scrapling.py  # Web fetching tools (3 tiers via Scrapling)
├── workspace/            # Agent identity + memory (auto-created)
│   ├── SOUL.md           # Agent personality + rules
│   ├── USER.md           # User context + preferences
│   ├── cron.json         # Scheduled job definitions
│   └── memory/           # Long-term memory
│       ├── knowledge.md  # Extracted knowledge from sessions
│       ├── learnings.md  # Extracted learnings (mistake → correction)
│       ├── index.json    # Reflection processing index
│       ├── memory.db     # SQLite vector + keyword search index
│       └── skills/       # Auto-extracted reusable Python skills
├── sessions/             # JSONL session transcripts (auto-created)
├── web/
│   └── index.html        # Chat web UI
├── start_gateway.sh      # Startup script
└── requirements.txt
```

## Operating Modes

### Bounded Mode (default)

Standard chat mode. The agent runs up to `max_loop_iterations` (default 15)
per message, executing tools as needed and returning a response.

### Deep Work Mode

Extended autonomous execution for complex multi-step tasks. Activated per-session
via WebSocket or per-request via the HTTP API.

Features:
- **Task decomposition** — Agent creates a task tree, works through subtasks one by one
- **Token budget** — Configurable limit with warning threshold
- **Auto-compaction** — When approaching the budget, conversation is archived to a markdown file and the context is trimmed, allowing work to continue
- **Nudge system** — If the agent produces text without using tools while tasks remain, the system injects a nudge to keep it on track

```bash
# HTTP API with deep work mode
curl -X POST http://localhost:8000/api/chat/my-session \
  -H "Content-Type: application/json" \
  -d '{"message": "Research and compile a report on...", "mode": "deep_work"}'
```

## Tools

| Tool | Description |
|------|-------------|
| `shell` | Execute shell commands (runs in workspace directory) |
| `read_file` | Read file contents (relative to workspace or absolute) |
| `write_file` | Write/append to files (auto-creates parent directories) |
| `list_directory` | List directory tree with depth control |
| `web_fetch` | Fast HTTP fetch with browser TLS impersonation |
| `web_fetch_js` | Full browser JS rendering for dynamic sites |
| `web_fetch_stealth` | Stealth browser with Cloudflare/bot detection bypass |
| `manage_tasks` | Create/update/complete/delete tasks (deep work mode) |
| `memory_search` | Search long-term memory via hybrid vector + keyword |

Auto-extracted skills from `workspace/memory/skills/` are also registered as tools at startup.

## Cron Scheduler

Define scheduled jobs in `workspace/cron.json`:

```json
[
  {
    "id": "daily-check",
    "name": "Daily status check",
    "schedule": "daily 09:00",
    "prompt": "Check system status and summarize any issues.",
    "session_id": "cron-daily-check",
    "enabled": true
  }
]
```

**Schedule formats:**
- `every 30m` / `every 2h` / `every 1d` — interval-based
- `daily 09:00` — daily at a specific UTC time
- `hourly :15` — every hour at minute 15
- `startup` — run once when the gateway starts

## Reflection Engine

On startup, the reflection engine scans completed sessions and uses the LLM to extract:

- **Knowledge** — Facts, API endpoints, configurations that were confirmed to work
- **Learnings** — Mistake → correction patterns to avoid repeating errors
- **Skills** — Reusable async Python functions, saved as `.py` files and auto-loaded as tools

Extracted items are stored in `workspace/memory/` and indexed for semantic search.

## HTTP API

```bash
# Chat (blocking)
curl -X POST http://localhost:8000/api/chat/my-session \
  -H "Content-Type: application/json" \
  -d '{"message": "List the files in my workspace"}'

# Chat with deep work mode
curl -X POST http://localhost:8000/api/chat/my-session \
  -H "Content-Type: application/json" \
  -d '{"message": "...", "mode": "deep_work"}'

# Status
curl http://localhost:8000/api/status

# List sessions
curl http://localhost:8000/api/sessions

# Session history
curl http://localhost:8000/api/sessions/my-session/history

# Session tasks (deep work)
curl http://localhost:8000/api/sessions/my-session/tasks

# Delete session
curl -X DELETE http://localhost:8000/api/sessions/my-session

# Token usage (range: today, 7d, 30d, all)
curl http://localhost:8000/api/usage?range=7d

# Recent activity
curl http://localhost:8000/api/activity?limit=50

# List available models
curl http://localhost:8000/api/models

# Switch model at runtime
curl -X POST http://localhost:8000/api/models/select \
  -H "Content-Type: application/json" \
  -d '{"provider": "lmstudio", "model_id": "your-model-id"}'

# Cron job status
curl http://localhost:8000/api/cron

# Trigger a cron job manually
curl -X POST http://localhost:8000/api/cron/daily-check/run

# Toggle a cron job on/off
curl -X POST http://localhost:8000/api/cron/daily-check/toggle

# Reload cron.json
curl -X POST http://localhost:8000/api/cron/reload
```

## WebSocket Protocol

Connect to `ws://localhost:8000/ws/{session_id}` and send/receive JSON.

**Client → Server:**
```json
{"type": "message", "content": "your message"}
{"type": "set_mode", "mode": "deep_work"}
{"type": "clear"}
{"type": "ping"}
```

**Server → Client:**
```json
{"type": "thinking", "iteration": 1}
{"type": "thinking", "iteration": 2, "max_iterations": 200, "tokens_used": 5000, "token_budget": 1500000, "task_summary": "..."}
{"type": "tool_call", "tool": "shell", "input": {"command": "ls"}, "tool_call_id": "..."}
{"type": "tool_result", "tool": "shell", "result_preview": "...", "duration_ms": 150, "success": true}
{"type": "text", "text": "Here are your files..."}
{"type": "task_update", "tasks": [...], "summary": "..."}
{"type": "mode_changed", "mode": "deep_work"}
{"type": "done", "text": "...", "iterations": 2, "model": "...", "usage": {...}, "mode": "bounded"}
{"type": "error", "message": "..."}
```

**Activity WebSocket** at `ws://localhost:8000/ws/activity` streams tool calls and results across all sessions in real time.

## Memory Search CLI

Test memory search independently:

```bash
# Default query
python memory_search.py

# Custom query
python memory_search.py "What API patterns have been discovered?"
```
