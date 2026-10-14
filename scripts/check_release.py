"""Smoke-test the built wheel outside the checkout, including bundled TUI assets."""
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main() -> None:
    wheels = list(Path(sys.argv[1]).resolve().glob("trx_ai-*.whl"))
    if len(wheels) != 1:
        raise ValueError("Expected exactly one TRX-AI wheel.")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert {"tui/app.tcss", "benchmarks/cases.jsonl", "progress.py"} <= names
        forbidden = {"agents.md", ".agents", ".codex", ".claude", "causmend"}
        assert not any(forbidden & set(name.lower().split("/")) for name in names)
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "installed"
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target), str(wheel)],
                       check=True, stdout=subprocess.DEVNULL)
        code = r'''
import asyncio
import os
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
os.environ['RD_USE_LOCAL_LLM'] = 'false'
os.environ['RD_EMBEDDINGS'] = 'false'
import main
from evaluation import load_dataset
from tui.app import TRXApp, ReviewCard
assert Path(main.__file__).resolve().is_relative_to(Path(sys.argv[1]))
assert len(load_dataset()) >= 50
assert main.main(['analyze', 'hello']) == 0
async def check_ui():
    app = TRXApp(demo=True)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert app.query(ReviewCard)
        assert app.query_one('#command').size.width > 50
asyncio.run(check_ui())
print('Wheel CLI, dataset, TUI assets, and demo passed.')
'''
        subprocess.run([sys.executable, "-I", "-c", code, str(target)], cwd=tmp, check=True)


if __name__ == "__main__":
    main()
