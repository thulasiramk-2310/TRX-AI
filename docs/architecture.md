# Architecture

TRX-AI follows a resilient, layered CLI architecture optimized for low-latency local code intelligence workflows.

## System Flow

1. User submits input in CLI.
2. Startup checks validate local LLM config and MCP graph availability (`.code-review-graph/graph.db`).
3. Hybrid intent detector (rules first, LLM fallback) classifies intent.
4. Router dispatches to analysis path:
   - Chat
   - Problem analysis
   - Code review/fix
5. Multi-agent reasoning executes:
   - Debug Agent
   - Improve Agent
   - Predict Agent
6. Local LLM (Ollama) generates structured content with bounded token budgets and retry control.
7. Formatter renders compact, sectioned output.
8. History/export modules persist reports.
9. Review cache stores results in memory + disk (`sessions/review_cache.json`) for warm starts.

## Diagram Description

Conceptual pipeline:

`User Input -> Intent Detection -> Multi-Agent Orchestration -> Local LLM + Rule Fallback -> Structured Output -> Report/History`

This design isolates intent routing, reasoning, generation, and rendering, improving maintainability and reliability.

## Reliability Layers

- Request retries with backoff
- Fast-fail LLM timeout defaults for quicker fallback
- Truncation-aware fixed-code recovery
- AST validation for generated patches
- Deterministic fallback reasoning and heuristic fixes
- Encoding-safe console rendering across terminals
- MCP graph state tagging in analysis status (`mcp_graph=ready|missing|empty|unreadable`)
- Directory exclusion and input-size caps for large-folder reviews
