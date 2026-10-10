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
    primary="#B5C4DC",
    secondary="#8894A8",
    accent="#B5C4DC",
    foreground="#D9DEE7",
    background="#0F1115",
    surface="#15181E",
    panel="#1C2027",
    success="#8ABB99",
    warning="#D6B578",
    error="#E88888",
    dark=True,
    variables={"block-cursor-background": "#B5C4DC", "footer-key-foreground": "#B5C4DC"},
)

THEMES = (NEON, HOLO, OPS)
