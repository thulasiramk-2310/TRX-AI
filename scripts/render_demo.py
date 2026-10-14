"""Render the README motion demo from the real Textual UI and labelled sample data.

Developer tooling only: requires ffmpeg, svglib, rl-renderPM, and project dependencies.
No model calls, session writes, generated fixes, or desktop recording are involved.
"""

from __future__ import annotations

import asyncio
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

from tui.app import TRXApp  # noqa: E402
from tui.demo import DEMO_STATUS  # noqa: E402
from tui.engine import SessionStats  # noqa: E402

BUILD = ROOT / "build" / "demo"
OUTPUT = ROOT / "assets" / "demo"
FONTS = ROOT / "report_assets" / "fonts"
WIDTH, HEIGHT, FPS, DURATION = 1280, 720, 30, 26
PAPER = "#ECEAE3"
INK = "#222830"
MUTED = "#626A71"
ACCENT = "#455E7A"
ORIGINAL = "def find_max(values):\n    return values[0]\n"
FIXED = 'def find_max(values):\n    if not values:\n        raise ValueError("empty list")\n    return max(values)\n'


class DemoEngine:
    """Only supplies status for screenshots; no production services are called."""

    stats = SessionStats()

    def status(self):
        return DEMO_STATUS


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


async def capture() -> dict[str, Image.Image]:
    images = {}
    app = TRXApp(engine=DemoEngine())
    app.workspace_name = "trx-ai"
    async with app.run_test(size=(124, 32)) as pilot:
        await pilot.pause()

        async def shot(name):
            await pilot.pause()
            path = Path(app.save_screenshot(filename=name + ".svg", path=str(BUILD)))
            images[name] = rasterize(path)

        await shot("ready")
        command = app.query_one("#command")
        text = "fix examples/find_max.py"
        for index in range(1, 9):
            command.value = text[:round(len(text) * index / 8)]
            await shot(f"typing-{index}")
        command.value = ""
        app._add_user(text)
        app._start_busy(text)
        await pilot.press("f3")
        for index, label in enumerate(("Reading · loading code", "Analyzer · finding bugs and risks",
                                       "Generator · writing review and code", "Critic · checking review quality (pass 1)")):
            app._show_progress(label)
            await shot(f"stage-{index}")
        await pilot.press("f3")
        result = {
            "response_mode": "analysis", "current_input": text,
            "review_target": "examples/find_max.py", "model_name": "qwen3:8b",
            "analysis_source": "llm", "debug_analysis": [
                "Empty input raises IndexError.", "The first value is not necessarily the maximum."],
            "improvements": ["Reject empty input explicitly; use max(values)."],
            "original_code": ORIGINAL, "fixed_code": FIXED,
            "system_status": ["steps=3", "critic_score=unavailable"], "guardrails": [],
        }
        app.show_result(result)
        await shot("review")
        app.query_one("#stream").scroll_end(animate=False)
        await shot("patch")
        app._confirm_fix(result)
        await shot("confirm")
        await pilot.press("escape")
    return images


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
    draw.text((60, 681), "ILLUSTRATIVE UI DEMO  /  EDITED PACING", font=font(11, mono=True), fill=MUTED)
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


def render_frame(t: float, images):
    if t < 2.7:
        return title_frame(t)
    if t < 6.8:
        local = t - 2.7
        key = "ready" if local < 0.6 else f"typing-{min(8, max(1, int((local - 0.6) * 3) + 1))}"
        zoom = 1 + 0.55 * ease((local - 0.3) / 1.8)
        return screen_frame(images[key], "Start with a file.", "01 / REQUEST", zoom, (0.25, 0.92))
    if t < 12.2:
        local = t - 6.8
        index = min(3, int(local / 1.35))
        zoom = 1.05 + 0.8 * ease(local / 2)
        return screen_frame(images[f"stage-{index}"], "Follow each review stage.", "02 / REVIEW", zoom, (0.91, 0.62))
    if t < 18.4:
        local = t - 12.2
        image = images["review"] if local < 2.8 else images["patch"]
        focus_y = 0.46 + 0.06 * ease((local - 2) / 2.5)
        return screen_frame(image, "Read the finding. Inspect the change.", "03 / DIFF", 1.24, (0.51, focus_y))
    if t < 22.4:
        local = t - 18.4
        zoom = 1.25 + 0.2 * ease(local / 2)
        return screen_frame(images["confirm"], "You decide when to save.", "04 / CONFIRM", zoom, (0.5, 0.5))
    return title_frame(t, closing=True)


def main():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("Install ffmpeg and add it to PATH before rendering the demo.")
    os.environ["RD_USE_LOCAL_LLM"] = "false"
    os.environ.pop("NO_COLOR", None)
    BUILD.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    images = asyncio.run(capture())
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
            for boundary in (2.7, 6.8, 12.2, 18.4, 22.4):
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
    render_frame(15, images).save(OUTPUT / "trx-demo-poster.png")
    for t in (1.5, 5, 9.5, 15, 20, 24):
        render_frame(t, images).save(BUILD / f"qa-{t:g}.png")
    print(f"Demo assets saved to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
