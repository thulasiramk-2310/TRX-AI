# Design Decisions

This document explains the key engineering decisions behind TRX-AI.

## 1. Hybrid Intent Detection

TRX-AI uses rule-first intent routing with LLM fallback.

Why:

- deterministic handling for greetings/commands/problem signals
- lower latency and fewer misroutes on common inputs
- graceful behavior when LLM is unavailable

## 2. Multi-Agent Reasoning

TRX-AI separates reasoning into focused agents:

- Debug: identify defects and failure modes
- Improve: suggest maintainability upgrades
- Predict: evaluate forward risk/performance trajectory

Why:

- cleaner cognitive decomposition
- better structured output quality
- easier extension and tuning per agent

## 3. Reliability-First Fallback Strategy

When local LLM fails or truncates output, TRX-AI applies:

- retries with backoff
- bounded timeouts to fail fast and return fallback output sooner
- deterministic section fallbacks
- heuristic fixed-code generation
- AST validation and repair loops
- MCP graph availability detection with graceful degradation

Why:

- avoid command-level hard failures
- keep `review` and `fix` operational under degraded conditions
- improve trust in CLI workflows

## 5. Latency and Throughput Controls

TRX-AI uses explicit guardrails for performance:

- review target character cap for folder reviews
- excluded directory set for recursive scans (`node_modules`, `.git`, `venv`, etc.)
- in-memory LRU caches for analysis and review paths
- optional disk-backed review cache to avoid repeated cold LLM calls

Why:

- lower startup and first-review latency
- avoid expensive scanning in large repositories
- reduce duplicate model token usage

## 4. Structured Output Contract

TRX-AI outputs stable sections:

- DEBUG
- IMPROVEMENTS
- PERFORMANCE
- FIX
- SUMMARY
- CONFIDENCE

Why:

- consistent readability
- easier export/report generation
- easier evaluation and automation
