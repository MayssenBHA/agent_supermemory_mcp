import asyncio
import os
import json
import requests

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient


# ─── REST HELPERS ─────────────────────────────────────────────────────────────

def fetch_all_memories(api_key: str) -> list:
    """Fetch only latest, non-forgotten memories via /v4/memories/list."""
    try:
        res = requests.post(
            "https://api.supermemory.ai/v4/memories/list",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"containerTags": ["sm_project_default"], "limit": 50},
            timeout=10,
        )
        if res.status_code != 200:
            print(f"⚠️  Could not load memories [{res.status_code}]: {res.text[:200]}")
            return []

        entries = res.json().get("memoryEntries", [])
        active = [
            m for m in entries
            if m.get("isLatest") is True and m.get("isForgotten") is False
        ]
        return active

    except Exception as e:
        print(f"⚠️  Memory fetch error: {e}")
        return []


def build_system_prompt(all_memories: list) -> str:
    """
    Build a fully autonomous system prompt.
    The agent reasons about memory management itself — no hardcoded rules.
    """

    if all_memories:
        lines = "\n".join(
            f"- (v{m.get('version','?')}) {m.get('memory') or str(m)}"
            for m in all_memories
        )
        memory_block = (
            "MEMORIES FROM PREVIOUS SESSIONS (already loaded):\n"
            + lines + "\n\n"
        )
    else:
        memory_block = "MEMORIES FROM PREVIOUS SESSIONS: None yet.\n\n"

    return (
        "You are a fully autonomous personal assistant agent with persistent memory, "
        "goal decomposition capabilities, and self-correction.\n\n"

        # ── SECTION 1: Memory Reasoning (not hardcoded rules) ─────────────────
        + memory_block
        + "MEMORY MANAGEMENT (reason this out yourself):\n"
        "You have access to `memory` and `recall` tools connected to Supermemory. "
        "You must autonomously decide:\n"
        "  • Whether a new fact shared by the user is worth persisting (is it personal, "
        "    recurring, or likely useful in future sessions?).\n"
        "  • Whether a fact already exists in your loaded memories before saving "
        "    (avoid duplicates by reasoning over the list above).\n"
        "  • Whether to use `recall` to search for something specific not in the list.\n"
        "  • When to update vs. append a memory (prefer updating stale facts).\n"
        "Do NOT follow rigid rules — use judgment based on context.\n\n"

        # ── SECTION 2: Goal Decomposition ─────────────────────────────────────
        + "GOAL DECOMPOSITION:\n"
        "When the user gives you a complex or multi-part request, you MUST:\n"
        "  1. Silently decompose the request into a numbered list of sub-goals.\n"
        "  2. Tackle each sub-goal sequentially, using tools as needed per step.\n"
        "  3. Track which sub-goals are complete, pending, or blocked.\n"
        "  4. After finishing all sub-goals, synthesize a unified final response.\n"
        "For simple requests, skip decomposition and respond directly.\n\n"

        # ── SECTION 3: Self-Correction Loop ───────────────────────────────────
        + "SELF-CORRECTION:\n"
        "If a tool call fails or returns an unexpected result, you MUST:\n"
        "  1. Diagnose the failure: wrong parameters? network issue? bad input?\n"
        "  2. Formulate a corrected approach (different params, different tool, "
        "     or a fallback strategy).\n"
        "  3. Retry up to 3 times with corrections before giving up.\n"
        "  4. If all retries fail, explain what you tried and what you know from memory.\n"
        "Never silently fail or say 'I cannot help' without attempting recovery.\n\n"

        # ── SECTION 4: General Behavior ───────────────────────────────────────
        + "GENERAL BEHAVIOR:\n"
        "  • Be proactive: if you notice something useful to remember, save it.\n"
        "  • Be transparent: briefly tell the user when you're decomposing a goal "
        "    or retrying a failed step.\n"
        "  • Prioritize answering from loaded memory before reaching for tools.\n"
    )


def list_memories_cli(api_key: str) -> None:
    """Print active memories to the console."""
    mems = fetch_all_memories(api_key)
    if not mems:
        print("\n  📭 No active memories stored yet.\n")
        return
    print(f"\n  📋 {len(mems)} active memories (isLatest=true, isForgotten=false):\n")
    for i, m in enumerate(mems):
        memory_text = m.get("memory") or str(m)
        version     = m.get("version", "?")
        print(f"  [{i+1}] (v{version}) {str(memory_text)[:120]}")
    print()


# ─── GOAL DECOMPOSITION DISPLAY ───────────────────────────────────────────────

def is_complex_request(user_input: str) -> bool:
    """
    Heuristic to decide whether to hint the agent to decompose.
    The agent itself reasons about this — this is just a UI hint.
    """
    complexity_signals = [
        len(user_input.split()) > 20,
        user_input.count("and") >= 2,
        any(kw in user_input.lower() for kw in [
            "then", "after that", "also", "additionally",
            "first", "second", "finally", "step"
        ]),
        "?" in user_input and len(user_input.split("?")) > 2,
    ]
    return sum(complexity_signals) >= 2


def print_thinking_header(user_input: str) -> None:
    if is_complex_request(user_input):
        print("\n🧩 Complex request detected — agent will decompose into sub-goals...\n")
    else:
        print("\n🤔 Thinking...\n")


# ─── SELF-CORRECTION WRAPPER ──────────────────────────────────────────────────

async def run_with_self_correction(agent, user_input: str, max_retries: int = 3) -> str:
    """
    Wraps agent.run() with an outer self-correction loop.
    On failure, it injects a diagnostic prompt so the agent can reason about
    what went wrong and try a different approach.
    """
    last_error = None
    attempt_context = user_input

    for attempt in range(1, max_retries + 1):
        try:
            response = await agent.run(attempt_context)

            # Check if the agent itself reported a tool failure in its response
            failure_signals = [
                "tool failed", "could not complete", "error occurred",
                "unable to", "failed to", "i apologize", "something went wrong"
            ]
            if any(signal in response.lower() for signal in failure_signals) and attempt < max_retries:
                print(f"⚠️  Agent reported an issue (attempt {attempt}/{max_retries}). "
                      f"Injecting diagnostic context...\n")
                attempt_context = (
                    f"[SELF-CORRECTION ATTEMPT {attempt}]\n"
                    f"Original request: {user_input}\n"
                    f"Your last response indicated a problem: \"{response[:300]}\"\n"
                    f"Diagnose what went wrong, choose a different approach or tool "
                    f"parameters, and try again. Be explicit about what you're changing."
                )
                last_error = response
                continue

            return response  # ✅ Success

        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                print(f"❌ Exception on attempt {attempt}/{max_retries}: {e}")
                print(f"🔄 Injecting correction context for retry {attempt + 1}...\n")
                attempt_context = (
                    f"[SELF-CORRECTION ATTEMPT {attempt}]\n"
                    f"Original request: {user_input}\n"
                    f"An exception occurred: {str(e)[:300]}\n"
                    f"Diagnose the failure, adjust your approach (different tool, "
                    f"different parameters, or answer from memory), and retry."
                )
            else:
                print(f"❌ All {max_retries} attempts failed. Last error: {e}")

    # Final fallback after all retries
    return (
        f"I attempted this {max_retries} times but encountered persistent issues. "
        f"Last known error: {last_error[:200] if last_error else 'unknown'}. "
        f"Here's what I can tell you from memory without tools: "
        f"[answering from loaded context only]"
    )


# ─── MAIN ─────────────────────────────────────────────────────────────────────

async def run_true_agent():
    load_dotenv()

    groq_key        = os.getenv("GROQ_API_KEY")
    supermemory_key = os.getenv("SUPERMEMORY_API_KEY")

    if not groq_key:
        print("❌ GROQ_API_KEY missing from .env"); return
    if not supermemory_key:
        print("❌ SUPERMEMORY_API_KEY missing from .env"); return

    # ── 1. Preload past memories ───────────────────────────────────────────────
    print("\n⏳ Loading memories from previous sessions...")
    all_memories = fetch_all_memories(supermemory_key)
    if all_memories:
        print(f"✅ {len(all_memories)} active memories loaded.\n")
    else:
        print("📭 No memories found — agent will bootstrap from scratch.\n")

    # ── 2. Build autonomous system prompt (agent reasons memory itself) ────────
    system_prompt = build_system_prompt(all_memories)

    # ── 3. MCP config ─────────────────────────────────────────────────────────
    mcp_config = {
        "mcpServers": {
            "supermemory": {
                "command": "npx",
                "args": [
                    "-y", "mcp-remote",
                    "https://mcp.supermemory.ai/mcp",
                    "--header",
                    f"Authorization: Bearer {supermemory_key}",
                    "--transport", "http-first",
                ],
            }
        }
    }

    print("🔌 Starting mcp-remote bridge to Supermemory MCP server...")
    client = MCPClient.from_dict(mcp_config)

    # ── 4. LLM ────────────────────────────────────────────────────────────────
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_key,
        temperature=0,
    )

    # ── 5. MCPAgent (max_steps raised for multi-step goal decomposition) ───────
    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=20,           # Raised: complex goals need more steps
        memory_enabled=True,
        system_prompt=system_prompt,
    )

    # ── 6. Chat loop ──────────────────────────────────────────────────────────
    print("\n🤖 True Autonomous Agent Ready")
    print("   ✅ Goal Decomposition  ✅ Self-Reasoning Memory  ✅ Self-Correction")
    print("Commands:  clear — reset chat   list — show memories   exit — quit")
    print("-" * 60)

    try:
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!"); break

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!"); break

            if user_input.lower() == "clear":
                agent.clear_conversation_history()
                print("✅ Conversation history cleared (memories preserved in Supermemory).")
                continue

            if user_input.lower() == "list":
                list_memories_cli(supermemory_key)
                continue

            print_thinking_header(user_input)

            # ── Self-correcting agent runner ───────────────────────────────
            response = await run_with_self_correction(agent, user_input, max_retries=3)
            print(f"Assistant: {response}")

    finally:
        if client and client.sessions:
            await client.close_all_sessions()


if __name__ == "__main__":
    asyncio.run(run_true_agent())