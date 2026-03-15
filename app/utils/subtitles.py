from __future__ import annotations

from typing import Any


def _format_timestamp(seconds: float, *, srt: bool) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    separator = "," if srt else "."
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}"


def _segment_text(segment: dict[str, Any]) -> str:
    speaker = segment.get("speaker")
    text = (segment.get("text") or "").strip()
    return f"[{speaker}] {text}" if speaker else text


def render_srt(segments: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{_format_timestamp(float(segment['start']), srt=True)} --> {_format_timestamp(float(segment['end']), srt=True)}",
                    _segment_text(segment),
                ]
            )
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_vtt(segments: list[dict[str, Any]]) -> str:
    lines = ["WEBVTT", ""]
    for segment in segments:
        lines.append(
            f"{_format_timestamp(float(segment['start']), srt=False)} --> {_format_timestamp(float(segment['end']), srt=False)}"
        )
        lines.append(_segment_text(segment))
        lines.append("")
    return "\n".join(lines)


def render_txt(segments: list[dict[str, Any]]) -> str:
    return "\n".join(_segment_text(segment) for segment in segments).strip() + "\n"


def render_tsv(segments: list[dict[str, Any]]) -> str:
    lines = ["start\tend\tspeaker\ttext"]
    for segment in segments:
        lines.append(
            "\t".join(
                [
                    str(segment.get("start", "")),
                    str(segment.get("end", "")),
                    str(segment.get("speaker", "") or ""),
                    (segment.get("text", "") or "").replace("\t", " ").replace("\n", " ").strip(),
                ]
            )
        )
    return "\n".join(lines) + "\n"
