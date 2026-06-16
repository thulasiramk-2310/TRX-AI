# Guardrails

TRX sends your code to an LLM and can write files based on what the model says. `guardrails.py`
sits between the model and your files, terminal and storage. All of it runs locally; nothing calls
an external service.

| Risk | Guardrail | Where |
|---|---|---|
| Secrets in reviewed code or chat (API keys, tokens, passwords, private keys, DB URLs) get saved to `sessions/` or uploaded to Supabase | `redact()` masks them as `[REDACTED]` at every write: session history, run history, vector memory, long-term memory, router memory, failure memory, review cache, event log, and cloud sync | `guardrails.redact_secrets`, each persistence module |
| Instructions hidden in the reviewed file ("ignore all previous instructions…") steer the reviewer | Every review prompt wraps the code in `<<<CODE … CODE>>>` with a rule saying it is untrusted data. Fake closing delimiters are neutralised. A warning shows in the review when such text is found. | `fence_untrusted_code`, `find_prompt_injection` |
| The model's "fix" adds dangerous code (`eval`, `os.system`, `shell=True`, `pickle.loads`, `rm -rf`, `curl \| sh`, …), breaks syntax, or drops most of the file | `check_fix` runs before you're asked to save. Blocking findings need you to type `force`; a reflexive `y` won't do. `watch` auto-fix never saves a blocked fix. | `check_fix`, `main.py` fix flow, `watcher.py` |
| Model output contains terminal escape codes (clear screen, change window title, write to clipboard via OSC 52) | Escape sequences and control characters are stripped before printing. Rich also strips them, so there are two layers. | `strip_terminal_controls`, `formatter.py` |

Existing protections that are still in place:
- Fixes are written to a separate `<name>_fixed.<ext>` file. The original file is never overwritten.
- Review targets are limited to an allowlist of file types, a maximum size, and excluded folders
  (`venv`, `.git`, …).

## Limits

- Secret detection uses patterns. It catches common key formats and `password = "…"`-style
  assignments, not every secret. Don't rely on it as your only protection.
- The dangerous-call check is lexical and only flags calls the fix *adds*. A call that was already
  in your file isn't blamed on the fix.
- The Python syntax check is the only syntax check; other languages get the call and size checks.
