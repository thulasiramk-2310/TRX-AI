"""Renders the TRX app headlessly and saves one SVG screenshot per theme."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from tui.app import TRXApp
from tui.themes import THEMES


async def shoot(theme: str, out_dir: Path, size: tuple[int, int]) -> Path:
    app = TRXApp(theme=theme, demo=True)
    async with app.run_test(size=size) as pilot:
        await pilot.pause(0.5)
        path = app.save_screenshot(filename=f"trx-{theme}.svg", path=str(out_dir))
    return Path(path)


async def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        print(await shoot(theme.name, out_dir, (150, 44)))


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1] if len(sys.argv) > 1 else ".")))
