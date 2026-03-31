"""Reality Debugger CLI entry point."""

from __future__ import annotations

import argparse
import difflib
import json
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.text import Text

from analyzer import RealityAnalyzer
from config import AppConfig
from formatter import OutputFormatter
from history import SessionHistory
try:
    from watcher import start_watcher
except Exception:  # pragma: no cover - optional runtime dependency
    start_watcher = None

LOGO_LINES = [
    "████████╗██████╗ ██╗  ██╗      █████╗ ██╗",
    "╚══██╔══╝██╔══██╗╚██╗██╔╝     ██╔══██╗██║",
    "   ██║   ██████╔╝ ╚███╔╝█████╗███████║██║",
    "   ██║   ██╔══██╗ ██╔██╗╚════╝██╔══██║██║",
    "   ██║   ██║  ██║██╔╝ ██╗     ██║  ██║██║",
    "   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝     ╚═╝  ╚═╝╚═╝",
]
ASCII_LOGO_LINES = [
    "TRX-AI",
]
RUN_HISTORY_PATH = Path("sessions") / "run_history.jsonl"
SUPPORTED_CODE_SUFFIXES = {
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".cc", ".cs",
    ".go", ".rs", ".swift", ".kt", ".php", ".sql", ".rb",
}


def run_cli(input_fn: Callable[[str], str] | None = None) -> None:
    console = Console()
    config = AppConfig.from_env()
    analyzer = RealityAnalyzer(config)
    formatter = OutputFormatter(
        console,
        typing_effect=config.typing_effect_enabled,
        typing_delay=config.typing_delay_seconds,
        ui_transitions=config.ui_transitions_enabled,
        ui_transition_delay=config.ui_transition_delay_seconds,
    )
    history = SessionHistory()

    mode = "debug"
    total_analyses = 0
    active_watchers: list[Any] = []

    prompt = input_fn or (lambda message: Prompt.ask(message))

    _print_startup(console)
    _print_dashboard(console, config, total_analyses, analyzer)
    _print_startup_warnings(console, config)

    while True:
        try:
            user_input = prompt("trx-ai >").strip()
        except (KeyboardInterrupt, EOFError):
            _shutdown_watchers(active_watchers)
            console.print("\nExiting Reality Debugger. Goodbye!", style="cyan")
            break

        if not user_input:
            _print_error(console, "Please enter a problem statement or a valid command.")
            continue

        command = user_input.lower()
        command_parts = user_input.split()

        if command in {"exit", "quit"}:
            _shutdown_watchers(active_watchers)
            console.print("Exiting Reality Debugger. Goodbye!", style="cyan")
            break

        if command == "help":
            _print_help(console)
            continue

        if command == "history":
            inputs = history.list_inputs()
            if not inputs:
                _print_error(console, "No history yet.")
            else:
                console.print("History:", style="cyan")
                for idx, item in enumerate(inputs, start=1):
                    console.print(f"{idx}. {item}")
            continue

        if command.startswith("save"):
            try:
                save_path = None
                if len(command_parts) > 1:
                    save_path = " ".join(command_parts[1:])
                saved_path = history.save(save_path)
                console.print(f"[OK] Session saved: {saved_path}", style="green")
            except OSError as exc:
                _print_error(console, f"Unable to save session: {exc}")
            continue

        if command.startswith("export"):
            if len(command_parts) >= 2 and command_parts[1].lower() == "compare":
                compare_entries = history.latest_analysis_entries(limit=2)
                if len(compare_entries) < 2:
                    _print_error(console, "No analysis available to export")
                    continue

                compare_name = "comparison_report.pdf"
                if len(command_parts) > 2:
                    compare_name = " ".join(command_parts[2:]).strip()

                first_entry, second_entry = compare_entries[0], compare_entries[1]
                first_structured = formatter.structured_text(first_entry["analysis"])
                second_structured = formatter.structured_text(second_entry["analysis"])

                try:
                    comparison_path = history.export_comparison_pdf_report(
                        compare_name,
                        first_input=str(first_entry.get("input", "")),
                        second_input=str(second_entry.get("input", "")),
                        mode=str(second_entry.get("mode", mode)),
                        first_structured_output=first_structured,
                        second_structured_output=second_structured,
                        first_label="Run 1",
                        second_label="Run 2",
                    )
                    console.print(f"[OK] Comparison report generated: {comparison_path}", style="green")
                except RuntimeError as exc:
                    _print_error(console, str(exc))
                continue

            latest = history.latest_entry()
            if not latest or not isinstance(latest.get("analysis"), dict):
                _print_error(console, "No analysis available to export")
                continue

            analysis = latest["analysis"]
            if analysis.get("response_mode") != "analysis":
                _print_error(console, "No analysis available to export")
                continue

            report_name = "report.txt"
            if len(command_parts) > 1:
                report_name = " ".join(command_parts[1:]).strip()

            structured = formatter.structured_text(analysis)
            try:
                if report_name.lower().endswith(".pdf"):
                    report_path = history.export_pdf_report(
                        report_name,
                        str(latest.get("input", "")),
                        str(latest.get("mode", mode)),
                        structured,
                    )
                else:
                    report_path = history.export_report(
                        report_name,
                        user_input=str(latest.get("input", "")),
                        mode=str(latest.get("mode", mode)),
                        structured_output=structured,
                    )
                console.print(f"[OK] Report generated: {report_path}", style="green")
            except RuntimeError as exc:
                _print_error(console, str(exc))
            continue

        if command.startswith("agents"):
            requested = [part.lower() for part in command_parts[1:]]
            if not requested:
                _print_error(console, "Usage: agents all OR agents debug improve predict")
                continue
            try:
                analyzer.set_active_agents(requested)
                _print_dashboard(console, config, total_analyses, analyzer)
            except ValueError as exc:
                _print_error(console, str(exc))
            continue

        if command in {"mode debug", "mode optimize", "mode predict"}:
            mode = command.split()[1]
            console.print(f"Fallback profile: {mode.upper()}", style="cyan")
            _print_dashboard(console, config, total_analyses, analyzer)
            continue

        if command.startswith("mode"):
            _print_error(console, "Invalid mode command. Use: mode debug, mode optimize, or mode predict")
            continue

        if command.startswith("review"):
            if len(command_parts) < 2:
                _print_error(console, "Usage: review <file.py | folder_path>")
                continue

            target = " ".join(command_parts[1:]).strip()
            try:
                code_blob = _load_review_target(target)
                result = analyzer.analyze_code_multi_agent(code_blob)
                result["original_code"] = code_blob[:20000]
                result["review_target"] = target
                total_analyses += 1
                formatter.render(result, mode, total_runs=total_analyses)
                history.add_entry(f"review {target}", "review", result)
                _append_run_history(f"review {target}", "review", result)
                _print_dashboard(console, config, total_analyses, analyzer)
            except FileNotFoundError:
                _print_error(console, f"File or folder not found: {target}")
            except IsADirectoryError:
                _print_error(console, f"Path is not a valid review target: {target}")
            except ValueError as exc:
                _print_error(console, str(exc))
            except OSError as exc:
                _print_error(console, f"Unable to read review target: {exc}")
            continue

        if command.startswith("fix"):
            if len(command_parts) < 2:
                _print_error(console, "Usage: fix <file.py>")
                continue

            target = " ".join(command_parts[1:]).strip()
            target_path = Path(target)
            try:
                if not target_path.exists() or not target_path.is_file():
                    raise FileNotFoundError(target)
                if target_path.suffix.lower() not in SUPPORTED_CODE_SUFFIXES:
                    raise ValueError(
                        "Unsupported format. Use a supported code file: "
                        + ", ".join(sorted(SUPPORTED_CODE_SUFFIXES))
                    )

                original_code = target_path.read_text(encoding="utf-8")
                if not original_code.strip():
                    raise ValueError(f"Empty file: {target_path}")

                result = analyzer.analyze_code_multi_agent(original_code)
                result["original_code"] = original_code[:20000]
                result["review_target"] = target
                total_analyses += 1
                formatter.render(result, mode, total_runs=total_analyses)
                history.add_entry(f"fix {target}", "fix", result)
                _append_run_history(f"fix {target}", "fix", result)

                fixed_code = str(result.get("fixed_code", "")).strip()
                if not fixed_code:
                    _print_error(console, "No fixed code was generated by the model.")
                    _print_dashboard(console, config, total_analyses, analyzer)
                    continue

                _print_fix_diff_preview(console, str(target_path), original_code, fixed_code)
                confirm = prompt("Apply fix? (y/n)").strip().lower()
                if confirm not in {"y", "yes"}:
                    console.print("Fix canceled.", style="yellow")
                    _print_dashboard(console, config, total_analyses, analyzer)
                    continue

                reason = ""
                final_insight = result.get("final_insight", [])
                if isinstance(final_insight, list) and final_insight:
                    reason = str(final_insight[0])
                elif isinstance(final_insight, str):
                    reason = final_insight

                fixed_path = apply_code_fix(
                    str(target_path),
                    fixed_code,
                    reason=reason,
                    original_code=original_code,
                )
                console.print(f"[OK] Fixed file saved as: {fixed_path}", style="green")
                _print_dashboard(console, config, total_analyses, analyzer)
            except FileNotFoundError:
                _print_error(console, f"File not found: {target}")
            except ValueError as exc:
                _print_error(console, str(exc))
            except OSError as exc:
                _print_error(console, f"Unable to apply fix: {exc}")
            continue

        if command.startswith("watch"):
            if len(command_parts) < 2:
                _print_error(console, "Usage: watch <folder>")
                continue

            target = " ".join(command_parts[1:]).strip()
            try:
                if start_watcher is None:
                    raise RuntimeError("watchdog is not available. Install with: pip install watchdog")

                watch_path = Path(target)
                if not watch_path.exists() or not watch_path.is_dir():
                    raise ValueError(f"Invalid watch path: {target}")

                auto_fix_answer = prompt("Apply fixes automatically? (y/n)").strip().lower()
                auto_fix = auto_fix_answer in {"y", "yes"}

                observer = start_watcher(
                    str(watch_path),
                    analyzer,
                    formatter,
                    auto_fix=auto_fix,
                    fix_writer=apply_code_fix,
                    debounce_seconds=1.0,
                )
                active_watchers.append(observer)
                console.print(f"[OK] Watching folder: {watch_path.resolve()}", style="green")
                if auto_fix:
                    console.print("[OK] Auto-fix enabled for watched changes.", style="green")
            except ValueError as exc:
                _print_error(console, str(exc))
            except RuntimeError as exc:
                _print_error(console, str(exc))
            except Exception as exc:
                _print_error(console, f"Unable to start watcher: {exc}")
            continue

        try:
            context = history.recent_context(limit=config.context_window_size)
            result = analyzer.analyze(user_input, mode=mode, past_context=context)
            if result.get("response_mode") == "analysis":
                total_analyses += 1
            formatter.render(result, mode, total_runs=total_analyses)
            history.add_entry(user_input, mode, result)
            _append_run_history(user_input, mode, result)
            _print_dashboard(console, config, total_analyses, analyzer)
        except ValueError as exc:
            _print_error(console, str(exc))
        except Exception as exc:
            _print_error(console, f"Unexpected error: {exc}")


def _print_error(console: Console, message: str) -> None:
    console.print(f"Error: {message}", style="red")


def _print_fix_diff_preview(
    console: Console,
    target_path: str,
    original_code: str,
    fixed_code: str,
) -> None:
    encoding = (getattr(getattr(console, "file", None), "encoding", "") or "").lower()
    terminal_codec = "cp1252" if encoding in {"cp1252", "windows-1252"} else ("ascii" if encoding == "ascii" else "")

    def safe_terminal_text(value: str) -> str:
        text = str(value)
        if not terminal_codec:
            return text
        return text.encode(terminal_codec, errors="replace").decode(terminal_codec, errors="replace")

    console.print(f"Fix preview for: {target_path}", style="cyan")
    console.print("Red = removed, Green = added", style="dim")

    diff_lines = list(
        difflib.unified_diff(
            original_code.splitlines(),
            fixed_code.splitlines(),
            fromfile=f"{target_path} (current)",
            tofile=f"{target_path} (fixed)",
            lineterm="",
            n=3,
        )
    )
    if not diff_lines:
        console.print("No code differences detected.", style="yellow")
        return

    max_preview_lines = 220
    for raw in diff_lines[:max_preview_lines]:
        line = safe_terminal_text(raw[:1000])
        text = Text(line)
        if line.startswith("+") and not line.startswith("+++"):
            text.stylize("green")
        elif line.startswith("-") and not line.startswith("---"):
            text.stylize("red")
        elif line.startswith("@@"):
            text.stylize("cyan")
        else:
            text.stylize("dim")
        console.print(text)
    if len(diff_lines) > max_preview_lines:
        console.print(f"... diff truncated ({len(diff_lines) - max_preview_lines} more lines)", style="dim")


def _load_review_target(target: str) -> str:
    path = Path(target)
    if not path.exists():
        raise FileNotFoundError(target)

    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_CODE_SUFFIXES:
            raise ValueError(
                "Unsupported format. Use a supported code file or a folder containing supported code files."
            )
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            raise ValueError(f"Empty file: {path}")
        return f"# FILE: {path.name}\n\n{content}"

    if not path.is_dir():
        raise IsADirectoryError(target)

    code_files = sorted(
        file_path for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_CODE_SUFFIXES
    )
    if not code_files:
        raise ValueError("No supported code files found in folder.")

    chunks: list[str] = []
    for file_path in code_files:
        content = file_path.read_text(encoding="utf-8")
        if not content.strip():
            continue
        relative_name = file_path.relative_to(path)
        chunks.append(f"# FILE: {relative_name}\n\n{content}")

    if not chunks:
        raise ValueError("All discovered supported code files are empty.")

    return "\n\n".join(chunks)


def _comment_style_for_path(path: Path) -> tuple[str, str]:
    ext = path.suffix.lower()
    if ext in {".py", ".sh", ".rb", ".yaml", ".yml", ".toml"}:
        return "# ", ""
    if ext in {".js", ".ts", ".java", ".c", ".cpp", ".cc", ".cs", ".go", ".rs", ".swift", ".kt", ".php"}:
        return "// ", ""
    if ext in {".sql"}:
        return "-- ", ""
    if ext in {".html", ".xml"}:
        return "<!-- ", " -->"
    return "# ", ""


def _build_fix_comment_header(
    *,
    destination_path: Path,
    original_code: str,
    fixed_code: str,
    reason: str,
    max_items: int = 8,
) -> str:
    """Builds a compact comment header that explains key code changes."""
    old_lines = original_code.splitlines()
    new_lines = fixed_code.splitlines()
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines)

    prefix, suffix = _comment_style_for_path(destination_path)

    def line(text: str) -> str:
        return f"{prefix}{text}{suffix}".rstrip()

    items: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        old_snippet = " | ".join(line.strip() for line in old_lines[i1:i2] if line.strip())[:120]
        new_snippet = " | ".join(line.strip() for line in new_lines[j1:j2] if line.strip())[:120]
        if not old_snippet and not new_snippet:
            continue
        items.append(line(f"- OLD: {old_snippet or '<none>'}"))
        items.append(line(f"+ NEW: {new_snippet or '<none>'}"))
        if len(items) >= max_items * 2:
            break

    header_lines = [
        line("TRX-AI CHANGE SUMMARY"),
        line(f"Reason: {reason.strip() or 'Automated reliability and correctness improvements.'}"),
    ]
    if items:
        header_lines.append(line("Changes:"))
        header_lines.extend(items)
    else:
        header_lines.append(line("Changes: Minor structural updates applied."))
    header_lines.append("")
    return "\n".join(header_lines)


def apply_code_fix(
    original_path: str,
    fixed_code: str,
    *,
    reason: str = "",
    original_code: str | None = None,
) -> Path:
    source = Path(original_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(original_path)

    destination = source.with_name(f"{source.stem}_fixed{source.suffix or '.txt'}")
    base_original = original_code if original_code is not None else source.read_text(encoding="utf-8")
    comment_header = _build_fix_comment_header(
        destination_path=destination,
        original_code=base_original,
        fixed_code=fixed_code,
        reason=reason,
    )
    destination.write_text(comment_header + fixed_code, encoding="utf-8")
    return destination


def _append_run_history(user_input: str, mode: str, result: dict[str, Any]) -> None:
    score = result.get("confidence")
    if score is None:
        if "confidence_score" in result:
            score = float(result.get("confidence_score", 0.0)) / 100.0
        else:
            score = float(result.get("intent_confidence", 0.0))
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input": user_input,
        "mode": mode,
        "output": str(result.get("analysis_text", result.get("chat_response", "")))[:2000],
        "score": float(score),
        "steps": result.get("system_status", []),
    }
    RUN_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUN_HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_benchmark_mode(runs: int = 20, debug: bool = False) -> None:
    from evaluation import EVAL_DATASET

    config = AppConfig.from_env()
    config.dev_mode = debug
    analyzer = RealityAnalyzer(config)

    successes = 0
    retries_total = 0.0
    latencies: list[float] = []

    for i in range(max(1, runs)):
        case = EVAL_DATASET[i % len(EVAL_DATASET)]
        started = time.perf_counter()
        result = analyzer.analyze_code_multi_agent(case.input_text)
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)

        score = float(result.get("confidence", 0.0))
        fixed = bool(str(result.get("fixed_code", "")).strip())
        if score > 0.0 and (fixed or "source=llm" in ",".join(result.get("system_status", []))):
            successes += 1

        status_joined = ",".join(str(s) for s in result.get("system_status", []))
        retries_total += float(config.local_llm_retries if "llm_unavailable" in status_joined else 1)

        _append_run_history(f"benchmark_case_{i+1}", "benchmark", result)

    total = max(1, runs)
    failures = total - successes
    avg_latency = sum(latencies) / total
    avg_retries = retries_total / total

    print("Model Performance:")
    print(f"- Success Rate: {(successes / total) * 100:.0f}%")
    print(f"- Avg Retries: {avg_retries:.1f}")
    print(f"- Failures: {failures}/{total}")
    print(f"- Avg Latency: {avg_latency:.2f}s")


def run_history_mode(limit: int = 20) -> None:
    if not RUN_HISTORY_PATH.exists():
        print("No run history found.")
        return
    lines = RUN_HISTORY_PATH.read_text(encoding="utf-8").splitlines()
    if not lines:
        print("No run history found.")
        return
    print("Run History:")
    for raw in lines[-max(1, limit):]:
        try:
            row = json.loads(raw)
        except ValueError:
            continue
        print(
            f"- {row.get('timestamp')} | mode={row.get('mode')} | "
            f"score={row.get('score')} | input={str(row.get('input',''))[:60]}"
        )


def run_analyze_mode(text: str, mode: str = "debug", debug: bool = False) -> None:
    config = AppConfig.from_env()
    config.dev_mode = debug
    analyzer = RealityAnalyzer(config)
    formatter = OutputFormatter(Console())

    result = analyzer.analyze(text, mode=mode, past_context=[])
    formatter.render(result, mode, total_runs=0)
    _append_run_history(text, mode, result)


def _shutdown_watchers(observers: list[Any]) -> None:
    for observer in observers:
        try:
            observer.stop()
            observer.join(timeout=2.0)
        except Exception:
            pass
    observers.clear()


def _print_dashboard(
    console: Console,
    config: AppConfig,
    total_analyses: int,
    analyzer: RealityAnalyzer,
) -> None:
    agents = ", ".join(analyzer.active_agents())
    typing = "ON" if config.typing_effect_enabled else "OFF"
    transition = "ON" if config.ui_transitions_enabled else "OFF"

    console.print()
    console.print(f"TRX-AI | Agents: {agents} | Runs: {total_analyses}")
    console.print(f"Model: {config.local_llm_model} | Typing: {typing} | UI: {transition}")
    console.print()


def _print_startup(console: Console) -> None:
    try:
        encoding = (getattr(getattr(console, "file", None), "encoding", None) or "").lower()
    except Exception:
        encoding = ""
    use_ascii = encoding in {"cp1252", "ascii"}
    logo_lines = ASCII_LOGO_LINES if use_ascii else LOGO_LINES

    for line in logo_lines:
        console.print(Align.center(f"[bold cyan]{line}[/bold cyan]"))
    console.print(Align.center("[dim]Reality Debugger[/dim]"))

    spinner = Spinner("dots", text="Initializing system...", style="cyan")
    with Live(Align.center(spinner), console=console, refresh_per_second=20, transient=True):
        time.sleep(0.8)

    console.clear()


def _print_help(console: Console) -> None:
    console.print("Commands:", style="cyan")
    console.print("help - Show available commands")
    console.print("history - Show previous inputs")
    console.print("save <path> - Save session")
    console.print("export <file> - Export last analysis")
    console.print("export compare <file> - Export comparison PDF from latest two analyses")
    console.print("review <code_file | folder_path> - Run multi-agent code review")
    console.print("fix <code_file> - Generate and save auto-fixed code as <name>_fixed.<ext>")
    console.print("watch <folder> - Watch Python files and auto-review on change")
    console.print("agents all | agents debug improve predict - Agent control")
    console.print("mode debug|optimize|predict - Fallback profile")
    console.print("exit | quit - Close Reality Debugger")
    console.print("Ctrl+C - Force quit the CLI")


def _print_startup_warnings(console: Console, config: AppConfig) -> None:
    env_missing = not Path(".env").exists()
    local_enabled = config.use_local_llm

    if not env_missing and local_enabled:
        return

    warning_lines: list[str] = []
    if env_missing:
        warning_lines.append("- .env file was not found in the project root.")
        warning_lines.append("- Copy values from .env.example to create a local .env file.")

    if not local_enabled:
        warning_lines.append("- RD_USE_LOCAL_LLM is disabled.")
        warning_lines.append("- Enable local AI by setting RD_USE_LOCAL_LLM=true in .env.")

    for line in warning_lines:
        wrapped = textwrap.fill(
            line,
            width=max(40, console.size.width - 4),
            break_long_words=False,
            break_on_hyphens=False,
        )
        console.print(wrapped, style="yellow")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRX-AI CLI")
    subparsers = parser.add_subparsers(dest="command")

    p_benchmark = subparsers.add_parser("benchmark", help="Run benchmark mode")
    p_benchmark.add_argument("--runs", type=int, default=20)
    p_benchmark.add_argument("--debug", action="store_true")

    p_history = subparsers.add_parser("history", help="Show persisted run history")
    p_history.add_argument("--limit", type=int, default=20)

    p_analyze = subparsers.add_parser("analyze", help="Analyze one input and exit")
    p_analyze.add_argument("text", type=str)
    p_analyze.add_argument("--mode", type=str, default="debug", choices=["debug", "optimize", "predict"])
    p_analyze.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    if args.command == "benchmark":
        run_benchmark_mode(runs=args.runs, debug=args.debug)
    elif args.command == "history":
        run_history_mode(limit=args.limit)
    elif args.command == "analyze":
        run_analyze_mode(text=args.text, mode=args.mode, debug=args.debug)
    else:
        run_cli()
