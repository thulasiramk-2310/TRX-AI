"""Render a concrete README walkthrough from actual local-model TUI captures.

Developer tooling only: requires ffmpeg, svglib, rl-renderPM, and project dependencies.
Run capture_demo.py first to record real inference and fix saving in an isolated workspace.
This editing step contacts no services and only reads those recorded screenshots.
"""

from __future__ import annotations

import json
from functools import lru_cache
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


BUILD = ROOT / "build" / "demo"
OUTPUT = ROOT / "assets" / "demo"
FONTS = ROOT / "report_assets" / "fonts"
WIDTH, HEIGHT, FPS, DURATION = 1280, 720, 30, 38
PAPER = "#ECEAE3"
INK = "#222830"
MUTED = "#626A71"
ACCENT = "#455E7A"
CAPTURE = {}


def rasterize(path: Path) -> Image.Image:
    from reportlab.graphics import renderPM
    from svglib.fonts import register_font
    from svglib.svglib import svg2rlg

    source = path.read_text(encoding="utf-8")
    rules = {name: dict(re.findall(r"([\w-]+)\s*:\s*([^;]+)", body))
             for name, body in re.findall(r"\.([\w-]+)\s*\{([^}]+)\}", source)}
    for weight in ("normal", "bold"):
        for style in ("normal", "italic"):
            register_font("DemoMono", str(FONTS / "IBMPlexMono-Regular.ttf"), weight=weight, style=style)
    root = ET.fromstring(source)

    def inline(element, inherited):
        styles = dict(inherited)
        for name in element.get("class", "").split():
            styles.update(rules.get(name, {}))
        if element.tag.endswith("text"):
            for key in ("fill", "font-size", "font-weight", "font-style"):
                if key in styles:
                    element.set(key, styles[key].strip())
            element.set("font-family", "DemoMono")
        # Rich screenshots already contain only visible terminal cells. svglib
        # applies nested text clipping with an incorrect transform; remove it.
        element.attrib.pop("clip-path", None)
        for child in element:
            inline(child, styles)

    inline(root, {})
    normalized = path.with_suffix(".normalized.svg")
    ET.ElementTree(root).write(normalized, encoding="utf-8", xml_declaration=True)
    png = path.with_suffix(".png")
    renderPM.drawToFile(svg2rlg(str(normalized)), str(png), fmt="PNG", dpi=144)
    with Image.open(png) as image:
        return image.convert("RGB")


def load_capture():
    record = json.loads((BUILD / "live-capture.json").read_text(encoding="utf-8"))
    if not record.get("original_preserved") or not record.get("fixed") or not all(
            record.get("verification", {}).get(key) for key in
            ("nonempty_max", "negative_max", "single_item", "empty_rejected")):
        raise ValueError("A successful live capture is required. Run scripts/capture_demo.py first.")
    images = {name: rasterize(BUILD / path) for name, path in record["screenshots"].items()}
    stages = record["stages"]
    if not stages:
        raise ValueError("No live progress screenshots were captured.")
    for index in range(4):
        selected = round(index * (len(stages) - 1) / 3)
        images[f"stage-{index}"] = images[f"live-stage-{selected}"]
    return images, record


@lru_cache(maxsize=32)
def font(size: int, *, mono=False, bold=False):
    name = "IBMPlexMono-Regular.ttf" if mono else "IBMPlexSans-SemiBold.ttf" if bold else "IBMPlexSans-Regular.ttf"
    return ImageFont.truetype(str(FONTS / name), size)


def ease(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * value * (value * (value * 6 - 15) + 10)


def camera(image: Image.Image, zoom: float, focus: tuple[float, float]) -> Image.Image:
    # A continuous camera crops the actual UI, preserving text and layout.
    target_w, target_h = 1160, 512
    ratio = target_w / target_h
    crop_w = min(image.width, image.height * ratio) / zoom
    crop_h = crop_w / ratio
    x = min(image.width - crop_w, max(0, focus[0] * image.width - crop_w / 2))
    y = min(image.height - crop_h, max(0, focus[1] * image.height - crop_h / 2))
    return image.resize((target_w, target_h), Image.Resampling.LANCZOS,
                        box=(x, y, x + crop_w, y + crop_h))


def screen_frame(image: Image.Image, caption: str, step: str, zoom=1.0, focus=(0.5, 0.5)):
    frame = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(frame)
    draw.text((60, 29), "TRX AI", font=font(22, bold=True), fill=INK)
    draw.text((1220, 34), step, font=font(14, mono=True), fill=MUTED, anchor="ra")
    draw.line((60, 66, 1220, 66), fill="#CACBC5", width=1)
    draw.rounded_rectangle((55, 87, 1225, 607), radius=10, fill="#D4D4CD")
    frame.paste(camera(image, zoom, focus), (60, 91))
    draw.text((60, 627), caption, font=font(29, bold=True), fill=INK)
    draw.text((60, 681), "LIVE QWEN3 CAPTURE  /  WAITING SHORTENED", font=font(11, mono=True), fill=MUTED)
    return frame


def title_frame(t: float, *, closing=False):
    frame = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(frame)
    draw.text((80, 62), "TRX AI  /  LOCAL ENGINEERING ASSISTANT", font=font(17, mono=True), fill=MUTED)
    draw.line((80, 106, 1200, 106), fill="#CACBC5", width=1)
    if closing:
        draw.text((80, 224), "Review the diff.", font=font(76, bold=True), fill=INK)
        draw.text((80, 314), "Keep the original.", font=font(76, bold=True), fill=INK)
        draw.text((84, 462), "python main.py", font=font(25, mono=True), fill=ACCENT)
    else:
        draw.text((80, 208), "TRX", font=font(136, bold=True), fill=INK)
        tagline = "Think. Reason. Execute."
        reveal = int(len(tagline) * ease((t - 0.4) / 1.3))
        draw.text((87, 384), tagline[:reveal], font=font(44), fill=INK)
        draw.text((87, 486), "Code review. Suggested fixes. Your terminal.", font=font(25), fill=MUTED)
    draw.text((80, 657), "github.com/thulasiramk-2310/TRX-AI", font=font(17, mono=True), fill=MUTED)
    return frame


def code_frame(*, after=False):
    frame = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(frame)
    draw.text((80, 42), "TRX AI  /  BROKEN CODE TO A SAVED FIX", font=font(17, mono=True), fill=MUTED)
    draw.text((80, 105), "The result: a maximum, not the first item." if after else
              "This function doesn't find the maximum.", font=font(41, bold=True), fill=INK)
    draw.rounded_rectangle((80, 188, 829, 552), radius=8, fill="#15191E")
    filename = "examples/find_max_fixed.py" if after else "examples/find_max.py"
    draw.text((101, 207), filename, font=font(17, mono=True), fill="#A8BACD")
    code = CAPTURE["fixed"] if after else CAPTURE["original"]
    lines = [line for line in code.splitlines() if not line.startswith("#")]
    for index, line in enumerate(lines):
        draw.text((101, 257 + index * 34), line, font=font(19, mono=True), fill="#E1E6EC")
    draw.text((874, 212), "EXAMPLE", font=font(17, mono=True), fill=MUTED)
    draw.text((874, 261), "[3, 1, 9]", font=font(25, mono=True), fill=INK)
    draw.text((874, 316), "returns 9" if after else "returns 3", font=font(29, bold=True), fill=INK)
    draw.text((874, 365), "Correct maximum." if after else "Expected: 9", font=font(22), fill=ACCENT)
    draw.text((874, 427), "Empty input:", font=font(21), fill=MUTED)
    draw.text((874, 464), "ValueError" if after else "IndexError", font=font(24, mono=True), fill=INK)
    draw.text((80, 600), "Saved a separate copy. Original unchanged." if after else
              "Ask TRX to review the file and suggest a fix.", font=font(29, bold=True), fill=INK)
    draw.text((80, 678), "ACTUAL CAPTURED SOURCE AND GENERATED CODE", font=font(11, mono=True), fill=MUTED)
    return frame


def render_frame(t: float, images):
    if t < 2:
        return title_frame(t)
    if t < 7:
        return code_frame()
    if t < 10.5:
        local = t - 7
        key = "ready" if local < 0.6 else f"typing-{min(8, max(1, int((local - 0.6) * 3) + 1))}"
        zoom = 1 + 0.55 * ease((local - 0.3) / 1.8)
        return screen_frame(images[key], "Request a fix for this file.", "01 / REQUEST", zoom, (0.25, 0.92))
    if t < 15.5:
        local = t - 10.5
        index = min(3, int(local / 1.25))
        zoom = 1.05 + 0.8 * ease(local / 2)
        return screen_frame(images[f"stage-{index}"],
                            f"Real Qwen3 review: {CAPTURE['elapsed_seconds']:.0f}s. Waiting shortened.",
                            "02 / LIVE REVIEW", zoom, (0.91, 0.62))
    if t < 20.5:
        return screen_frame(images["review"], "TRX identifies the incorrect return and empty-input crash.",
                            "03 / FINDINGS", 1.15, (0.51, 0.32))
    if t < 25.5:
        return screen_frame(images["patch"], "Inspect the actual generated change.", "04 / DIFF", 1.18, (0.51, 0.62))
    if t < 28.5:
        local = t - 25.5
        zoom = 1.25 + 0.2 * ease(local / 2)
        return screen_frame(images["confirm"], "Confirm before TRX saves the copy.", "05 / CONFIRM", zoom, (0.5, 0.5))
    if t < 31.5:
        return screen_frame(images["saved"], "Saved: examples/find_max_fixed.py", "06 / SAVED", 1.18, (0.51, 0.84))
    if t < 35.5:
        return code_frame(after=True)
    return title_frame(t, closing=True)


def main():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("Install ffmpeg and add it to PATH before rendering the demo.")
    os.environ["RD_USE_LOCAL_LLM"] = "false"
    os.environ.pop("NO_COLOR", None)
    BUILD.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    global CAPTURE
    images, CAPTURE = load_capture()
    keys = ("captured_at", "model", "elapsed_seconds", "original", "fixed", "output",
            "original_preserved", "stages", "verification")
    (OUTPUT / "capture.json").write_text(json.dumps({key: CAPTURE[key] for key in keys}, indent=2), encoding="utf-8")
    mp4 = OUTPUT / "trx-demo.mp4"
    command = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
               "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for index in range(FPS * DURATION):
            t = index / FPS
            frame = render_frame(t, images)
            # Brief dissolves soften scene changes without obscuring the reading holds.
            for boundary in (2, 7, 10.5, 15.5, 20.5, 25.5, 28.5, 31.5, 35.5):
                if boundary <= t < boundary + 0.3:
                    frame = Image.blend(render_frame(boundary - 0.001, images), frame,
                                        ease((t - boundary) / 0.3))
            process.stdin.write(frame.tobytes())
            if index % (FPS * 3) == 0:
                print(f"Rendered {t:.0f}s / {DURATION}s", flush=True)
    finally:
        process.stdin.close()
        code = process.wait()
    if code:
        raise SystemExit(f"Video encoding failed ({code}).")
    palette = BUILD / "palette.png"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                    "fps=10,scale=800:-1:flags=lanczos,palettegen=max_colors=128", str(palette)], check=True)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(mp4), "-i", str(palette),
                    "-lavfi", "fps=10,scale=800:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                    "-loop", "0", str(OUTPUT / "trx-demo.gif")], check=True)
    render_frame(23, images).save(OUTPUT / "trx-demo-poster.png")
    for t in (1, 4, 9, 13, 18, 23, 27, 30, 33, 37):
        render_frame(t, images).save(BUILD / f"qa-{t:g}.png")
    print(f"Demo assets saved to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
