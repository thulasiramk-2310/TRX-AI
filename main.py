"""Reality Debugger CLI entry point."""

from __future__ import annotations

import textwrap
import time
from pathlib import Path
from typing import Any, Callable

from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.prompt import Prompt
from rich.spinner import Spinner

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

        if command == "exit":
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
                total_analyses += 1
                formatter.render(result, mode, total_runs=total_analyses)
                history.add_entry(f"review {target}", "review", result)
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
                if target_path.suffix.lower() != ".py":
                    raise ValueError("Unsupported format. Use a .py file for fix.")

                original_code = target_path.read_text(encoding="utf-8")
                if not original_code.strip():
                    raise ValueError(f"Empty file: {target_path}")

                result = analyzer.analyze_code_multi_agent(original_code)
                total_analyses += 1
                formatter.render(result, mode, total_runs=total_analyses)
                history.add_entry(f"fix {target}", "fix", result)

                fixed_code = str(result.get("fixed_code", "")).strip()
                if not fixed_code:
                    _print_error(console, "No fixed code was generated by the model.")
                    _print_dashboard(console, config, total_analyses, analyzer)
                    continue

                confirm = prompt("Apply fix? (y/n)").strip().lower()
                if confirm not in {"y", "yes"}:
                    console.print("Fix canceled.", style="yellow")
                    _print_dashboard(console, config, total_analyses, analyzer)
                    continue

                fixed_path = apply_code_fix(str(target_path), fixed_code)
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
            _print_dashboard(console, config, total_analyses, analyzer)
        except ValueError as exc:
            _print_error(console, str(exc))
        except Exception as exc:
            _print_error(console, f"Unexpected error: {exc}")


def _print_error(console: Console, message: str) -> None:
    console.print(f"Error: {message}", style="red")


def _load_review_target(target: str) -> str:
    path = Path(target)
    if not path.exists():
        raise FileNotFoundError(target)

    if path.is_file():
        if path.suffix.lower() != ".py":
            raise ValueError("Unsupported format. Use a .py file or a folder containing .py files.")
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            raise ValueError(f"Empty file: {path}")
        return f"# FILE: {path.name}\n\n{content}"

    if not path.is_dir():
        raise IsADirectoryError(target)

    py_files = sorted(path.rglob("*.py"))
    if not py_files:
        raise ValueError("No .py files found in folder.")

    chunks: list[str] = []
    for file_path in py_files:
        content = file_path.read_text(encoding="utf-8")
        if not content.strip():
            continue
        relative_name = file_path.relative_to(path)
        chunks.append(f"# FILE: {relative_name}\n\n{content}")

    if not chunks:
        raise ValueError("All discovered .py files are empty.")

    return "\n\n".join(chunks)


def apply_code_fix(original_path: str, fixed_code: str) -> Path:
    source = Path(original_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(original_path)

    destination = source.with_name(f"{source.stem}_fixed.py")
    destination.write_text(fixed_code, encoding="utf-8")
    return destination


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
    console.print("save [path] - Save session")
    console.print("export <file> - Export last analysis")
    console.print("export compare [file] - Export comparison PDF from latest two analyses")
    console.print("review <file.py | folder_path> - Run multi-agent code review")
    console.print("fix <file.py> - Generate and save auto-fixed code as <name>_fixed.py")
    console.print("watch <folder> - Watch Python files and auto-review on change")
    console.print("agents all | agents debug improve predict - Agent control")
    console.print("mode debug|optimize|predict - Fallback profile")
    console.print("exit - Close Reality Debugger")


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
    run_cli()
