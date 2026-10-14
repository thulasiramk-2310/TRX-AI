# TRX-AI

**Think. Reason. Execute.**

TRX-AI is a CLI-first, hybrid AI assistant for structured debugging, code review, and safe auto-fix workflows using local LLMs and resilient fallbacks.

Built for real engineering loops: fast intent routing, multi-agent review, structured output, observability, and deterministic runtime behavior.

---

## Why TRX

- `Hybrid intelligence`: Rule-first + semantic routing + feedback learning.
- `Engineering-first UX`: Built for terminal-native developer workflows.
- `Reliable by default`: Fallbacks, retries, cache TTL, and runtime status signals.
- `Actionable outputs`: Structured review sections, not vague chat blobs.
- `Showcase-ready`: Clean architecture, measurable behavior, and testable components.

---

## Feature Highlights

- ⚡ **Hybrid Intent Routing**: Route requests to general or code workflows with confidence signals.
- 🧠 **Multi-Agent Code Pipeline**: Analyzer, Generator, Critic, and Fixer stages.
- 🛡️ **Fallback Safety**: Graceful behavior when LLM or graph context is degraded.
- 📈 **Observability**: Structured logs, latency metrics, cache hit/miss, breaker state.
- 🗂️ **Session & Export Tools**: History, save/export, comparison reports.
- 🧪 **Smoke + Unit Testing**: `main.py test`, `smoke_e2e.py`, and CI-friendly assertions.

---

## How It Works

TRX receives user input, classifies intent, chooses the best execution path, then returns either conversational output or structured analysis with metadata.

1. Detect intent with rule + semantic signals.
2. Route to general assistant flow or code analysis flow.
3. Apply cache, fallback, and observability instrumentation.
4. Return output with runtime metadata (`mcp_query_status`, `circuit_breaker_state`, `cache_hit`).

---

## Architecture

### System Architecture Flow

```mermaid
flowchart LR
    U[User Input] --> R[Hybrid Router]
    R -->|General| G[General Assistant Flow]
    R -->|Code| C[Code Analysis Flow]
    G --> O[Unified Output Formatter]
    C --> O
    O --> CLI[CLI / Reports / History]
    O --> OBS[Metrics + Structured Logs]
```

### Intent Routing Pipeline

```mermaid
flowchart TD
    A[Input Text] --> B[Rule Signals]
    A --> C[Semantic Signals]
    B --> D[Confidence Merge]
    C --> D
    D --> E{Route Decision}
    E -->|General| F[General QA Handler]
    E -->|Code| G[Multi-Agent Review]
    D --> H[Router Feedback Memory]
    H --> E
```

### Multi-Agent Pipeline

```mermaid
flowchart LR
    I[Code / Prompt] --> AN[Analyzer]
    AN --> GE[Generator]
    GE --> CR[Critic]
    CR -->|Score OK| FX[Fixer/Finalize]
    CR -->|Needs revision| GE
    FX --> OUT[Structured Review Output]
```

---

## CLI Demo

```bash
# Start TRX
python main.py

# Ask general questions
trx-ai > hi
trx-ai > what is data warehouse

# Review code
trx-ai > review examples/dsa_test.py

# Generate fixed code
trx-ai > fix examples/dsa_test.py

# Show runtime status
trx-ai > status

# Run built-in smoke summary
python main.py test

# Run machine-readable end-to-end smoke report
python smoke_e2e.py --disable-llm
```

---

## Commands

- `help` - Show available commands
- `history` - Show session inputs
- `save <path>` - Save session JSON
- `export <file>` - Export latest analysis report (`.txt` or `.pdf`)
- `export compare <file>` - Export comparison PDF from latest two analyses
- `agents all | agents debug improve predict` - Control active agents
- `mode debug|optimize|predict` - Set fallback profile
- `review <code_file | folder_path>` - Run multi-agent code review
- `fix <code_file>` - Generate fixed code and ask before saving
- `watch <folder>` - Auto-review changed code files
- `python main.py test` - Run TRX smoke test summary
- `python smoke_e2e.py --disable-llm` - Emit JSON pass/fail matrix report
- `exit | quit` - Close CLI

Every command also works with a slash, e.g. `/review app.py` or `/help`.

### Interactive prompt

On a real terminal, the prompt uses [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/):

- **Tab** completes commands, sub-commands (`mode o` → `optimize`), and file paths after `review`, `fix`, `watch`.
- **Up/Down** recall earlier inputs. History is saved to `sessions/cli_history.txt`, so it survives restarts.
- **Grey suggestions** appear as you type, based on history. Press **→** to accept one.
- **Status** sits dimmed at the right edge of the prompt line: model, mode, and how long the last answer took. There is no toolbar and no reserved space, so the screen doesn't jump.
- **Ctrl+D** exits. Pasting multi-line code works.

When input is piped (scripts, CI), TRX falls back to a plain prompt automatically.

### Output

Answers print into a scrolling transcript; the screen is never cleared.

- **Chat answers** render as Markdown, followed by one muted line: model · time · source (`llm`, `built-in knowledge`, or `rule fallback`).
- **Code reviews** render as plain sections (Issues / Fixes / Performance / Summary) with a confidence score. There are no boxes, so copied text stays clean. See [docs/ui.md](docs/ui.md) for the design rules and the user feedback behind them, and [docs/guardrails.md](docs/guardrails.md) for safety checks (secret redaction, fix vetting, prompt-injection fencing).
- **Fixes** show as a colored diff against your file (`+added -removed`), or "no code changes suggested". You never get a full copy of an unchanged file.
- Reviews longer than 2,500 characters use only the input prefix and explicitly report that limit. Whole-file fixes are withheld for these inputs; review a smaller file or function to generate a fix.
- **Structured logs** go to `sessions/trx_events.jsonl` instead of the terminal. Set `RD_STRUCTURED_LOGS=stdout` to print them, or `false` to turn them off.

---

## Clean Output Example

```text
[TRX RESPONSE]

💬 Response:
Here’s the root cause: the login check does not validate empty tokens.

Insight:
Add a guard clause before parsing and return a typed auth error.

Confidence: High
Runtime:
- intent: code
- mcp_query_status: ACTIVE
- circuit_breaker_state: CLOSED
- cache_hit: false
```

---

## Reliability & Observability

- Rule-first routing with semantic enhancement
- Retry/backoff for local LLM calls
- Deterministic in-memory cache TTL
- Review results expire with the configured cache TTL; temporary model failures are retried rather than cached. Chat caches and conversation context are scoped by user and personality, and guest retrieval searches only guest memories.
- Structured runtime events (`intent_route`, `response_runtime`)
- First-class breaker and MCP status in outputs
- Graceful degraded-mode fallbacks

---

## Evaluation & Testing

- Unit tests, lint, and smoke tests (all run in CI on every push/PR):

```bash
pip install -r requirements-dev.txt
pytest --cov
ruff check .
python smoke_e2e.py --disable-llm --min-score 10
```

- Code-review benchmark (66 labelled cases across 6 languages; needs a running local LLM):

```bash
trx eval                          # full run with plain-prompt baseline
trx eval --no-baseline --limit 10 # quick check
trx eval --category security      # one category
```

Every rate is reported with its sample size and a 95% Wilson confidence interval,
next to a "chance floor" (what a generic, code-agnostic review would score).
Results land in `reports/eval/` as text + JSON with model, git commit, and dataset hash.

- JSON smoke report includes:
- overall score out of 10
- category summaries
- pass/fail matrix per check

See:
- [docs/evaluation.md](docs/evaluation.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/design.md](docs/design.md)

---

## Setup

1. Install (Python 3.10+), from the repo root:

```bash
pip install -e .                 # core, provides the `trx` command
pip install -e ".[semantic]"     # optional: embedding router + FAISS vector memory
pip install -e ".[dev]"          # optional: pytest, coverage, ruff
```

`pip install -r requirements.txt` still works if you prefer running `python main.py`.

2. Configure `.env`:

```env
RD_USE_LOCAL_LLM=true
LOCAL_LLM_URL=http://localhost:11434/api/generate
LOCAL_LLM_MODEL=qwen3:8b
HF_REQUEST_TIMEOUT=120
HF_MAX_NEW_TOKENS=600
HF_TEMPERATURE=0.3
RD_ASSISTANT_MODE=auto
RD_CACHE_SIZE=120
RD_CACHE_TTL_SECONDS=300
RD_DEBUG_CACHE=false
RD_REVIEW_LOGS=false
```

3. Run:

```bash
trx              # full-screen workspace (F2 switches theme)
trx --classic    # classic scrolling CLI (also used automatically when output is piped)
```

The full-screen app has a compact workspace header, a keyboard-driven sidebar, and a wide
conversation area with code reviews and diffs. Press F3 to open the diagnostics inspector
for confidence, response time, safety checks, and a live stage trace. The thinking loader displays elapsed time and
the current generation stage even with the inspector closed. Reviews run Analyzer → Generator
→ Critic sequentially; Fixer appears when a revision is attempted. Chat, cache hits, fallback,
and safety checks report their own stages. `fix <file>` opens a confirmation dialog; a fix blocked by guardrails needs
`force` typed before it can be saved.

Use Tab and arrow keys to navigate the sidebar, Enter to select, Ctrl+L to focus
the command bar, and Escape to cancel a fix confirmation. The app hides side panels
in narrower terminals to keep the conversation readable. History and benchmarks
are available through the classic CLI. The default theme uses neutral surfaces and one muted
accent; the terminal controls the font (Cascadia Code and JetBrains Mono both work well).

---

## Project Structure

```text
chatcli/
|-- main.py              # CLI entry point (`trx`)
|-- analyzer.py          # RealityAnalyzer orchestration
|-- intent_rules.py      # rule-based intent signals
|-- rule_engine.py       # deterministic fallback reasoning
|-- llm_client.py        # local Ollama client with retry/backoff
|-- formatter.py
|-- watcher.py
|-- history.py
|-- config.py
|-- evaluation.py
|-- semantic_scoring.py
|-- smoke_e2e.py
|-- benchmarks/cases.jsonl
|-- examples/
|-- pyproject.toml
|-- tests/
|-- docs/
|-- assets/
|-- README.md
|-- LICENSE
```

---

## License

MIT - see [LICENSE](LICENSE).

