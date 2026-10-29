"""Deterministic recursive character chunking with reproducible boundaries."""
from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass
from typing import List, Optional

import tiktoken


@dataclass
class Chunk:
    text: str
    chunk_index: int
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    token_count: int
    checksum: str  # of text for reproducibility


# Preferred split points (higher priority first)
SEPARATORS = [
    "\n\n\n",
    "\n\n",
    "\n",
    ". ",
    "! ",
    "? ",
    "; ",
    ", ",
    " ",
    "",
]


def _token_count(text: str, enc) -> int:
    return len(enc.encode(text))


def _split_text(text: str, separators: List[str]) -> List[str]:
    if not separators:
        return [text]
    sep = separators[0]
    remaining = separators[1:]
    if sep == "":
        return list(text)
    parts = text.split(sep)
    result = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            part = part + sep
        if part:
            result.append(part)
    # Further split oversized pieces
    final = []
    for p in result:
        if len(p) > 2000:  # hard safety
            final.extend(_split_text(p, remaining))
        else:
            final.append(p)
    return final


def recursive_chunk(
    text: str,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    page_map: Optional[List[tuple[int, int, int]]] = None,
    encoding_name: str = "cl100k_base",
) -> List[Chunk]:
    """
    Deterministic recursive splitter.
    page_map: list of (char_start, char_end, page_num) covering the full document.
    """
    if not text or not text.strip():
        return []

    try:
        enc = tiktoken.get_encoding(encoding_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    # Normalize whitespace lightly but keep structure
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 4+ newlines to 2
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    pieces = _split_text(text, SEPARATORS)
    chunks: List[Chunk] = []
    current: List[str] = []
    current_len = 0
    global_char = 0
    piece_offsets: List[tuple[str, int]] = []  # (piece, start_char)

    # Rebuild with offsets
    pos = 0
    for p in pieces:
        piece_offsets.append((p, pos))
        pos += len(p)

    def flush(start_char: int, end_char: int, body: str, idx: int):
        body = body.strip()
        if not body:
            return
        # Resolve pages
        p_start, p_end = 1, 1
        if page_map:
            pages = set()
            for cs, ce, pg in page_map:
                if ce > start_char and cs < end_char:
                    pages.add(pg)
            if pages:
                p_start, p_end = min(pages), max(pages)
        tc = _token_count(body, enc)
        ck = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
        chunks.append(
            Chunk(
                text=body,
                chunk_index=idx,
                page_start=p_start,
                page_end=p_end,
                char_start=start_char,
                char_end=end_char,
                token_count=tc,
                checksum=ck,
            )
        )

    buf: List[str] = []
    buf_start = 0
    buf_tokens = 0
    idx = 0

    for piece, start in piece_offsets:
        ptok = _token_count(piece, enc)
        if buf and buf_tokens + ptok > chunk_size:
            body = "".join(buf)
            end = start
            flush(buf_start, end, body, idx)
            idx += 1
            # Overlap: keep trailing tokens
            if chunk_overlap > 0 and body:
                # Simple char-based overlap approximation
                overlap_chars = min(len(body), int(chunk_overlap * 4))  # ~4 chars/token
                tail = body[-overlap_chars:]
                buf = [tail]
                buf_start = end - len(tail)
                buf_tokens = _token_count(tail, enc)
            else:
                buf = []
                buf_start = start
                buf_tokens = 0
        if not buf:
            buf_start = start
        buf.append(piece)
        buf_tokens += ptok

    if buf:
        body = "".join(buf)
        flush(buf_start, buf_start + len(body), body, idx)

    return chunks


def build_page_map_from_pages(pages: List[str]) -> List[tuple[int, int, int]]:
    """Given list of page texts (index 0 = page 1), return (char_start, char_end, page)."""
    page_map = []
    pos = 0
    for i, pg in enumerate(pages):
        # Add page break marker length if we join with \n\n
        start = pos
        end = pos + len(pg)
        page_map.append((start, end, i + 1))
        pos = end + 2  # for "\n\n"
    return page_map
