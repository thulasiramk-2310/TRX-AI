# Terminal UI design

`trx` opens the full-screen app in an interactive terminal. `trx --classic` uses the
scrolling CLI described below; piped output also uses the classic CLI.

## Full-screen app

- The sidebar provides Chat, Code review, Generate fix, and Help shortcuts. Use Tab
  to focus it, arrow keys to move, and Enter to select. Review and fix fill the
  command bar so you can add the target path before submitting.
- Ctrl+L focuses the command bar, F2 cycles the three themes, and Ctrl+Q quits.
- The default workspace theme uses neutral surfaces and a single muted accent.
  Reviews and patches use quiet separators; conversation text uses the full available width.
- F3 or `inspect` toggles the diagnostics inspector. It starts hidden so the
  transcript has more room. Resizing preserves your preference.
- During a request a subtle animated loader shows the operation and elapsed time.
  The activity strip shows the current stage and recent stages even with diagnostics
  closed. F3 shows the stage trace. Reviews run Analyzer, Generator, and Critic in
  sequence; Fixer appears only when a revision is attempted. Chat uses its own
  generation path, and cache hits and rule fallback are labelled explicitly.
  Stage notifications contain no prompts, code, or hidden model reasoning. A
  second request stays in the command bar while the current operation finishes.
- Fixes show a diff before the save dialog. Escape or Cancel dismisses the dialog
  without writing a file; blocked fixes require typing `force` to enable Save.
- Below 120 columns the inspector is hidden; below 90 columns the sidebar is
  also hidden. All commands remain available in the command bar.
- History, evaluation, and other classic CLI operations remain available through
  `trx --classic`; the full-screen sidebar lists its implemented actions.

## Classic CLI

The classic CLI prints into the terminal's normal scrollback, like a shell. These rules come from public
feedback on Claude Code, Codex CLI, Gemini CLI and Copilot CLI. Please keep them when you change
`formatter.py`, `cli_input.py` or the loop in `main.py`.

## What users complain about, and what TRX does instead

| Complaint | Cause | TRX rule |
|---|---|---|
| The terminal jumps to the bottom while you scroll up to read ([claude-code#18299](https://github.com/anthropics/claude-code/issues/18299), [#34845](https://github.com/anthropics/claude-code/issues/34845)) | Constant redraws. One analysis measured thousands of scroll events per second from full-screen re-renders ([write-up](https://namiru.ai/blog/claude-code-s-terminal-flickering-700-upvotes-9-months-still-broken)) | Each answer is rendered once and written in one go (`OutputFormatter._print`). There is no animated spinner. While the model works, one static "working" line is written, then erased once. |
| Flicker during output on Windows ([claude-code#16939](https://github.com/anthropics/claude-code/issues/16939)) | Live regions repainting many times per second | There are no live regions. |
| Copied text includes border characters `│ ┃` ([copilot-cli#4116](https://github.com/github/copilot-cli/issues/4116)) | Answers drawn inside boxes | Answers have no boxes. Sections are a coloured heading plus indented lines, and the fixed code has no line numbers. |
| The screen shifts up and down at the prompt ([prompt_toolkit#1249](https://github.com/prompt-toolkit/python-prompt-toolkit/issues/1249)) | Reserved completion-menu space, a bottom toolbar, and a popup that opens on every keystroke | `reserve_space_for_menu=0`, no bottom toolbar, and Tab-only readline-style completion. Status sits in a dim right-hand prompt on the same line. |
| Lost scrollback history ([codex#35335](https://github.com/openai/codex/issues/35335)) | Alternate screen buffer or clearing the screen | The classic CLI never clears the screen and never uses the alternate screen. |
| Blank or garbled lines after resizing the window | Lines padded with spaces up to the terminal width | Trailing padding is stripped before writing. |

What users praise: output that stays in the native scrollback, so select, scroll and find
behave the way they always do. See [xli](https://github.com/vitalops/xli) for the same approach.

## Output layout

```
● Review app.py   confidence 86%

  Issues
  ✗ Line 2: find_max returns arr[0] without checking for an empty list.

  Fixes
  ✓ Guard empty input: if not arr: raise ValueError("empty list").

  Suggested change  +7 -1
     def find_max(arr):
    -    return arr[0]
    +    if not arr:
    ...

  qwen3:8b · 6.4s · llm · graph: 3 related modules
```

- Use one accent colour (green). Use colour only where it means something: red for issues,
  green for fixes, yellow for performance, blue for the summary.
- Use single-width glyphs only (`● ✗ ✓ ↯ →`). Emoji such as ⚡ are double-width in some terminals
  and break alignment. Cp1252 consoles get ASCII fallbacks automatically.
- The grey meta line under every answer says what produced it: `llm`, `built-in knowledge` or
  `rule fallback (LLM unavailable)`. Never label a fallback as `llm`.
- Graph-context lines ("Impacted module: …") are context, not findings. They are summarised in
  the meta line.

## Input

- `❯` prompt, and history saved in `sessions/cli_history.txt`.
- Grey suggestions from history appear as you type; press `→` to accept one.
- Tab completes commands, sub-commands and file paths. Every command also works as `/command`.
- Structured logs go to `sessions/trx_events.jsonl`, never into the transcript.
  `RD_STRUCTURED_LOGS=stdout` prints them instead.
