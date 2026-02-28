# agent_computer 🦞

A stripped-down, Python-based agent framework inspired by OpenClaw's architecture.
Model-agnostic via **OpenRouter** — use Claude, GPT, Gemini, Llama, DeepSeek, or
any of hundreds of models through a single API.

## Core Concepts

- **Gateway** — Single FastAPI process handling WebSocket + HTTP connections
- **Agent Runtime** — The agentic loop: context assembly → LLM → tool execution → reply
- **Sessions** — Stateful conversations persisted as JSONL, with serial execution
- **Tools** — Modular, registerable capabilities the LLM can call
- **Workspace** — Markdown files (SOUL.md, USER.md) that define agent identity

## Quick Start

```bash
pip install -r requirements.txt

# Get your key at https://openrouter.ai/keys
export OPENROUTER_API_KEY=sk-or-...

# Start the gateway
python gateway.py
```

Then open `http://localhost:8000` for the web UI, or connect via WebSocket at
`ws://localhost:8000/ws/{session_id}`.

## Switching Models

Edit `config.json` and change `model_id` to any OpenRouter model:

```json
"model": {
  "model_id": "anthropic/claude-sonnet-4-5-20250514"
}
```

Some popular options:
- `anthropic/claude-sonnet-4-5-20250514` — great balance of speed + quality
- `anthropic/claude-opus-4-6` — best reasoning
- `openai/gpt-4o` — OpenAI's flagship
- `google/gemini-2.5-pro-preview` — Google's best
- `deepseek/deepseek-r1` — strong open-source reasoning
- `meta-llama/llama-4-maverick` — Meta's latest

Full model list: https://openrouter.ai/models

## Project Structure

```
agent_computer/
├── gateway.py          # FastAPI gateway (entry point)
├── agent.py            # Agent runtime + agentic loop (OpenRouter/OpenAI SDK)
├── session.py          # Session management + JSONL persistence
├── tool_registry.py    # Tool registration and execution
├── context.py          # System prompt assembly from workspace files
├── config.py           # Configuration loader
├── config.json         # Agent + model configuration
├── tools/
│   └── __init__.py     # Built-in tools (shell, files, web fetch)
├── workspace/          # Agent identity files
│   ├── SOUL.md         # Agent personality + rules
│   └── USER.md         # User context
├── sessions/           # JSONL session transcripts (auto-created)
├── web/
│   └── index.html      # Chat web UI
└── requirements.txt
```

## HTTP API

```bash
# Chat (blocking)
curl -X POST http://localhost:8000/api/chat/my-session \
  -H "Content-Type: application/json" \
  -d '{"message": "List the files in my workspace"}'

# Status
curl http://localhost:8000/api/status

# List sessions
curl http://localhost:8000/api/sessions

# Delete session
curl -X DELETE http://localhost:8000/api/sessions/my-session
```

## WebSocket Protocol

Connect to `ws://localhost:8000/ws/{session_id}` and send/receive JSON:

**Client → Server:**
```json
{"type": "message", "content": "your message"}
{"type": "clear"}
{"type": "ping"}
```

**Server → Client:**
```json
{"type": "thinking", "iteration": 1}
{"type": "tool_call", "tool": "shell", "input": {"command": "ls"}, "tool_call_id": "..."}
{"type": "tool_result", "tool": "shell", "result_preview": "..."}
{"type": "text", "text": "Here are your files..."}
{"type": "done", "text": "...", "iterations": 2, "model": "anthropic/claude-sonnet-4-5-20250514"}
{"type": "error", "message": "..."}
```

## Architecture (What Maps to OpenClaw)

| OpenClaw | agent_computer |
|----------|----------|
| Gateway (Node.js process) | `gateway.py` (FastAPI) |
| Agent Loop (intake → LLM → tools → reply) | `agent.py` |
| Session Management (JSONL + serial exec) | `session.py` |
| Workspace files (SOUL.md, USER.md) | `workspace/` |
| Tool system | `tool_registry.py` + `tools/` |
| Multi-channel adapters | WebSocket + HTTP REST |
| Model-agnostic (`openclaw.json`) | OpenRouter (`config.json`) |
| Skills / ClawHub | (future — add as needed) |

## What's Deliberately Left Out

These can be layered in when you're ready:

- **Multi-agent routing** — multiple isolated agents with bindings
- **Skills/plugin system** — modular capability loading
- **Heartbeat/cron** — proactive scheduled behavior
- **Memory compaction** — summarizing old context
- **Sandboxing** — Docker isolation, tool deny lists
- **Additional channels** — Telegram, Discord, Slack adapters
