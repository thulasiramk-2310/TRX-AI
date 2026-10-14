# README motion demo

The 26-second silent walkthrough uses screenshots rendered by the real TRX Textual app.
It shows a typed fix request, the sequential review-stage inspector, a finding and code diff,
and the save confirmation. Sample responses and edited pacing are labelled in every UI scene;
the clip does not measure model speed or represent a live Ollama session.

The intentionally incorrect `examples/find_max.py` fixture is included so the request shown
in the video can be tried in TRX. Actual model responses and generation times will vary.

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
python scripts/render_demo.py
```

The renderer uses the bundled IBM Plex fonts. It writes intermediate screenshots to ignored
`build/demo/`, and final MP4, GIF, and poster files to `assets/demo/`. It contacts no model or
graph services, records no desktop content, and saves no source-code fixes or sessions.

Review `build/demo/qa-*.png` and the MP4 after changing the app layout or the camera timeline.
Use `ffprobe assets/demo/trx-demo.mp4` to inspect encoding and duration.
