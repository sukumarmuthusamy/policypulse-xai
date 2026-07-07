"""Run a single agent query against the configured LLM and local FAISS index."""

from __future__ import annotations

import sys

from app.agents.orchestrator import run_agent
from app.agents.schemas import AgentRequest


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python scripts/verify_agent.py "Your policy question here"')
        return 1

    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print("Error: query must not be empty.", file=sys.stderr)
        return 1

    response = run_agent(AgentRequest(query=query))

    print("Agent verification complete.")
    print(f"  Trace ID: {response.trace.trace_id}")
    print(f"  Latency: {response.trace.latency_ms} ms")
    print(f"  Tool calls: {len(response.trace.tool_calls)}")
    print(f"  Retrieved chunks: {len(response.trace.retrieved_chunks)}")
    print()
    print("Answer:")
    print(response.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
