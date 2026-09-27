"""Normalize extracted document text before chunking."""

from __future__ import annotations

import re

_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_FORM_FEED = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def preprocess_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _FORM_FEED.sub(" ", cleaned)
    cleaned = _HYPHEN_BREAK.sub(r"\1\2", cleaned)
    cleaned = _MULTI_SPACE.sub(" ", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
    cleaned = _MULTI_NEWLINE.sub("\n\n", cleaned)
    return cleaned.strip()
