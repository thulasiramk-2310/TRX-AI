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
- deterministic section fallbacks
- heuristic fixed-code generation
- AST validation and repair loops

Why:

- avoid command-level hard failures
- keep `review` and `fix` operational under degraded conditions
- improve trust in CLI workflows

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
