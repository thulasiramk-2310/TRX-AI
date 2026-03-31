# Architecture

TRX-AI follows a resilient, layered CLI architecture optimized for local code intelligence workflows.

## System Flow

1. User submits input in CLI.
2. Hybrid intent detector (rules first, LLM fallback) classifies intent.
3. Router dispatches to analysis path:
   - Chat
   - Problem analysis
   - Code review/fix
4. Multi-agent reasoning executes:
   - Debug Agent
   - Improve Agent
   - Predict Agent
5. Local LLM (Ollama) generates structured content.
6. Formatter renders compact, sectioned output.
7. History/export modules persist reports.

## Diagram Description

Conceptual pipeline:

`User Input -> Intent Detection -> Multi-Agent Orchestration -> Local LLM -> Structured Output -> Report/History`

This design isolates intent routing, reasoning, generation, and rendering, improving maintainability and reliability.

## Reliability Layers

- Request retries with backoff
- Truncation-aware fixed-code recovery
- AST validation for generated patches
- Deterministic fallback reasoning and heuristic fixes
- Encoding-safe console rendering across terminals
