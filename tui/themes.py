"""TRX colour themes for the full-screen app. Each is a complete Textual theme."""

from __future__ import annotations

from textual.theme import Theme

NEON = Theme(
    name="trx-neon",
    primary="#22D3EE",
    secondary="#A78BFA",
    accent="#F472B6",
    foreground="#E2E8F0",
    background="#05070D",
    surface="#0A0F1E",
    panel="#111833",
    success="#34D399",
    warning="#FBBF24",
    error="#FB7185",
    dark=True,
    variables={"block-cursor-background": "#22D3EE", "footer-key-foreground": "#22D3EE"},
)

HOLO = Theme(
    name="trx-holo",
    primary="#7FE7FF",
    secondary="#5EEAD4",
    accent="#C4F1FF",
    foreground="#D6F3FF",
    background="#020B12",
    surface="#04141F",
    panel="#082335",
    success="#7DFFCF",
    warning="#FFE08A",
    error="#FF9AA2",
    dark=True,
    variables={"block-cursor-background": "#7FE7FF", "footer-key-foreground": "#7FE7FF"},
)

OPS = Theme(
    name="trx-ops",
    primary="#F2B84B",
    secondary="#4FD1A5",
    accent="#8FB6FF",
    foreground="#D7E0EE",
    background="#070B14",
    surface="#0D1524",
    panel="#14213A",
    success="#4FD1A5",
    warning="#F2B84B",
    error="#FF6B6B",
    dark=True,
    variables={"block-cursor-background": "#F2B84B", "footer-key-foreground": "#F2B84B"},
)

THEMES = (NEON, HOLO, OPS)
