# agent_supermemory_mcp

A fully autonomous personal assistant with persistent memory, goal decomposition, and self-correction. Built with `mcp_use`, `LangChain`, `Groq (LLaMA 3.3 70B)`, and `Supermemory`.

---

## Features

- **🧩 Goal Decomposition** — Breaks complex multi-part requests into sequential sub-goals and executes them one by one
- **🧠 Self-Reasoning Memory** — Autonomously decides what to save, update, or recall — no hardcoded rules
- **🔄 Self-Correction** — Diagnoses tool failures and retries up to 3 times with corrected parameters
- **💾 Persistent Memory** — Remembers facts across sessions via Supermemory

---

## Requirements

```
python >= 3.10
node >= 18 (for mcp-remote via npx)
```

Install dependencies:

```bash
pip install mcp-use langchain-groq python-dotenv requests
```

---

## Setup

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
SUPERMEMORY_API_KEY=your_supermemory_api_key
```

---

## Run

```bash
python app.py
```

---

## Commands

| Command | Description |
|---------|-------------|
| `list`  | Show all stored memories |
| `clear` | Reset conversation history (memories preserved) |
| `exit`  | Quit the agent |

---

## Architecture

```
User Input
    │
    ▼
Goal Decomposition (complex requests split into sub-goals)
    │
    ▼
MCPAgent (LLaMA 3.3 70B via Groq)
    │
    ├── recall / memory tools → Supermemory API (via mcp-remote)
    │
    └── Self-Correction Loop (up to 3 retries on failure)
    │
    ▼
Response
```

---

## Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq — LLaMA 3.3 70B Versatile |
| Agent framework | `mcp_use` MCPAgent |
| Memory backend | Supermemory (REST + MCP) |
| MCP transport | `mcp-remote` (stdio ↔ HTTP) |
| Environment | Python + `python-dotenv` |
