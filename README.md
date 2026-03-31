# TRX-AI (Reality Debugger)

TRX-AI is a CLI-based AI assistant for structured reasoning and code review.
It supports hybrid intent detection, local LLM inference (Ollama), multi-agent analysis,
and clean terminal output with Rich.

## Features

- Hybrid intent routing (rule-first + LLM fallback)
- Multi-agent analysis pipeline:
  - Debug
  - Improve
  - Predict
  - Code review mode
- Local LLM support (default: `qwen3:8b` via Ollama)
- CLI commands:
  - `help`
  - `history`
  - `save [path]`
  - `export <file.txt|file.pdf>`
  - `export compare [file.pdf]`
  - `agents all | agents debug improve predict`
  - `mode debug|optimize|predict`
  - `review <file.py | folder_path>`
  - `fix <file.py>`
  - `watch <folder>`
  - `exit`
- Structured output sections:
  - DEBUG
  - IMPROVEMENTS
  - PERFORMANCE
  - FIX
  - SUMMARY
  - CONFIDENCE
- Evaluation + benchmarking module (`evaluation.py`) with baseline comparison

## Project Structure

```text
chatcli/
|-- main.py
|-- analyzer.py
|-- formatter.py
|-- history.py
|-- watcher.py
|-- evaluation.py
|-- config.py
|-- dsa_test.py
|-- README.md
|-- .env.example (optional template)
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables in `.env` (or system env):

- `RD_USE_LOCAL_LLM=true`
- `LOCAL_LLM_URL=http://localhost:11434/api/generate`
- `LOCAL_LLM_MODEL=qwen3:8b`
- `HF_REQUEST_TIMEOUT=120`
- `HF_MAX_NEW_TOKENS=600`
- `HF_TEMPERATURE=0.3`

4. Ensure Ollama is running and model is available:

```bash
ollama run qwen3:8b
```

5. Start CLI:

```bash
python main.py
```

## Usage Examples

```text
trx-ai > review dsa_test.py
trx-ai > fix dsa_test.py
trx-ai > watch .
trx-ai > export quick_report.txt
```

## Evaluation / Benchmark

Run full evaluation suite:

```bash
python evaluation.py
```

Outputs:

- Accuracy
- Fix Quality
- Average Response Time
- Completeness
- TRX vs baseline comparison
- `evaluation_report.txt`

## Notes

- `fix <file.py>` never overwrites original source; it writes `<name>_fixed.py`.
- `review <folder>` scans Python files recursively.
- In some terminals (cp1252), Unicode characters are safely downgraded to avoid crashes.
