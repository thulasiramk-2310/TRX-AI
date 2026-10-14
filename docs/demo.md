# README motion demo

The 38-second silent walkthrough starts with a concrete bug: `find_max([3, 1, 9])` returns `3`
instead of `9`, and empty input crashes. It then shows a real local Qwen3 8B review through the
TRX engine and Textual UI, the actual generated diff, save confirmation, a visible saved-file
message, and the resulting code. The fixed copy returns `9` and explicitly rejects empty input.

The intentionally incorrect `examples/find_max.py` fixture is included so the request shown
in the video can be tried in TRX. The captured review took **117.09 seconds**. Waiting is shortened
and labelled in the video; this single run is not a performance benchmark. Future model responses
and generation times will vary. No review response or saved-file result is invented for this clip.

The captured source, generated code, model, time, stages, and checks are recorded in
[`assets/demo/capture.json`](../assets/demo/capture.json). The source remained unchanged and the
fixed copy passed four concrete checks: a normal list, negative numbers, a single item, and
empty-input rejection. The capture tool permits only a restricted pure-function AST for this
fixture before running these checks in a separate Python process; it rejects other generated code.

The README uses an animated GIF for inline playback and links to the full 720p, 30 fps H.264
MP4. The animation uses eased camera zooms, short dissolves, and static reading holds. It has
no audio, flashing effects, or downloaded third-party footage. A static poster is also supplied.

References used for direction:

- [Anthropic's Claude Code launch](https://www.anthropic.com/news/claude-3-7-sonnet)
  and [official introduction](https://www.youtube.com/watch?v=AJpK3YTTKZ4).
- [Screen Studio zooms](https://preview.screen.studio/guide/adding-editing-zooms)
  and [animation controls](https://preview.screen.studio/guide/animations).

To regenerate from the repository root, install FFmpeg on PATH and the optional render tools
in your development environment:

```bash
python -m pip install -r requirements-dev.txt svglib rl-renderPM
```

For a new capture, start Ollama with the configured model available, then run:

```bash
python scripts/capture_demo.py
python scripts/render_demo.py
```

The capture runs actual inference and confirmation/save operations in a temporary workspace,
with cloud memory, embeddings, and disk review caching disabled. It captures the app's rendered
screens rather than desktop content. Original and saved files are verified before cleanup.

The renderer uses bundled IBM Plex fonts and requires a successful verified capture. It writes
intermediate screenshots to ignored `build/demo/`, and final MP4, GIF, poster, and provenance
files to `assets/demo/`. The editing/rendering step contacts no model or graph services.

Review `build/demo/qa-*.png` and the MP4 after changing the app layout or the camera timeline.
Use `ffprobe assets/demo/trx-demo.mp4` to inspect encoding and duration.
