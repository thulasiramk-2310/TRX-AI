# TRX-AI

> Built with reliability, explainability, and structured reasoning at its core.
>
> **"A resilient multi-agent AI system for structured debugging and code intelligence."**

![CI](https://img.shields.io/github/actions/workflow/status/thulasiramk-2310/TRX-AI/ci.yml?branch=main&label=CI)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/github/license/thulasiramk-2310/TRX-AI)

TRX-AI is a CLI-first, local-LLM powered code intelligence system that combines hybrid intent detection and multi-agent reasoning to produce structured, actionable outputs for debugging and code review.

## Features

- Hybrid intent detection (rule + LLM)
- Multi-agent reasoning (Debug, Improve, Predict)
- Code review + auto-fix
- Local LLM (Ollama)
- Structured CLI UI
- Evaluation + benchmarking
- Fallback reliability system

## Demo

```bash
trx-ai > review dsa_test.py
trx-ai > fix dsa_test.py
python evaluation.py
```

## Architecture

Pipeline:

`User -> Intent -> Agents -> LLM -> Structured Output`

- User input enters the CLI
- Hybrid intent detection classifies the request
- Specialized agents generate focused reasoning
- Local LLM synthesizes deep analysis
- Formatter renders structured output sections

For detailed architecture notes, see [docs/architecture.md](docs/architecture.md).

## Evaluation

TRX-AI includes a benchmark dataset and evaluation engine in `evaluation.py`.

Metrics:

- Accuracy
- Fix Quality
- Avg Response Time
- Completeness

Graph:

![Accuracy Graph](./assets/trx_ai_accuracy_graph.png)

Screenshot:

![Evaluation Output](./assets/evaluation_output.png)

Detailed methodology is documented in [docs/evaluation.md](docs/evaluation.md).

## Reliability

TRX-AI is built to remain useful under imperfect model conditions.

- Retry logic with backoff for local LLM calls
- AST validation for generated fixed code
- Auto-repair and continuation when output is truncated
- Deterministic fallback analysis when LLM is unavailable
- Heuristic fallback fixed-code generation for continuity

## Installation

```bash
pip install -r requirements.txt
```

## Quickstart

```bash
python main.py
```

## Usage

```bash
trx-ai > help
trx-ai > review <file.py | folder_path>
trx-ai > fix <file.py>
trx-ai > watch <folder>
trx-ai > export quick_report.txt
```

## Project Structure

```text
trx-ai/
├── analyzer.py
├── formatter.py
├── evaluation.py
├── main.py
├── tests/
│   └── test_trx_ai.py
├── docs/
│   ├── architecture.md
│   ├── evaluation.md
│   └── design.md
├── assets/
│   ├── trx_ai_accuracy_graph.png
│   └── evaluation_output.png
├── README.md
├── CONTRIBUTING.md
├── requirements.txt
├── LICENSE

```

## Roadmap

- Stronger section-level semantic scoring
- Expanded deterministic patch templates
- More evaluation cases across security/performance domains
- Better visualization and dashboard export
- Plugin-ready architecture for custom agents

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
