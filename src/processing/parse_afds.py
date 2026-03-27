import re
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import config

# sections worth embedding
KEEP_SECTIONS = {"SYNOPSIS", "SHORT TERM", "LONG TERM", "DISCUSSION", "NEAR TERM"}

# match the timestamp line
HEADER_TIMESTAMP_RE = re.compile(
    r"^\d{3,4} [AP]M \w{2,3}T \w{3} \w{3} \d{1,2} \d{4}",
    re.MULTILINE,
)

# section delimiter
SECTION_RE = re.compile(r"^\.([A-Z ]+?)(?:\s*/[^/]*/)?\.{3}", re.MULTILINE)


def _strip_header(text):
    """Remove the AFOS product header and return just the body text."""
    match = HEADER_TIMESTAMP_RE.search(text)
    if match:
        return text[match.end():].strip()
    # if that doesn't work, skip to after AFDBGM
    idx = text.find("AFDBGM")
    if idx != -1:
        rest = text[idx + len("AFDBGM"):]
        return rest.strip()
    return text.strip()


def _extract_sections(text):
    """For structured AFDs, pull out the meteorological narrative sections."""
    splits = list(SECTION_RE.finditer(text))
    if not splits:
        return None

    chunks = []
    for i, match in enumerate(splits):
        name = match.group(1).strip()
        if not any(name.startswith(s) for s in KEEP_SECTIONS):
            continue
        start = match.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        chunk = text[start:end].strip()
        # cut off trailing and section delimiter
        chunk = re.sub(r"\s*&&.*", "", chunk, flags=re.DOTALL).strip()
        chunk = re.sub(r"\n{3,}", "\n\n", chunk)
        if chunk:
            chunks.append(chunk)

    return "\n\n".join(chunks) if chunks else None


def _extract_freeform(text):
    """
    For old-format AFDs with no section headers, take the body and drop
    any trailing .BGM advisory lines and forecaster initials.
    """
    # drop .BGM short local advisory summaries
    text = re.sub(r"\n\.BGM.*", "", text, flags=re.DOTALL).strip()
    # drop lone forecaster initials
    text = re.sub(r"\n[A-Z]{2,6}\s*$", "", text).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def extract_body(text):
    body = _strip_header(text)

    sections = _extract_sections(body)
    if sections:
        return sections

    return _extract_freeform(body)


def load_afds(afd_dir=None):
    """
    Load all AFDs from disk, returning a dict of date_str -> text.
    If multiple AFDs exist for the same day, we keep the earliest one.
    """
    if afd_dir is None:
        afd_dir = config.RAW_AFD_DIR
    afd_dir = Path(afd_dir)

    records = {}  # date_str -> (length, body)

    for f in sorted(afd_dir.glob("*.txt")):
        date_str = f.name[:10]

        try:
            raw = f.read_text(errors="replace")
            body = extract_body(raw)
            if len(body) > 300:
                # keep the longest AFD of the day
                if date_str not in records or len(body) > records[date_str][0]:
                    records[date_str] = (len(body), body)
        except Exception as e:
            print(f"  skipping {f.name}: {e}")

    return {date: body for date, (_, body) in records.items()}
