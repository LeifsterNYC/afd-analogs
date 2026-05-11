import argparse
import json
import subprocess
import sys
import os
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import config

# extracted frames live alongside processed/, one JSON file per date
OUT_DIR = config.PROCESSED_DIR / "frames"

PROMPT = """You are extracting structured event frames from a National Weather Service Area Forecast Discussion (AFD).

For each distinct weather event mentioned in the AFD, return one JSON object with these fields:
- event_type: one of [snow, rain, freezing_rain, sleet, thunderstorm, wind, fog, heat, cold, lake_effect, other]
- intensity: one of [light, moderate, heavy, severe, scattered, isolated]
- region: short string describing the area (e.g. "western NY", "Poconos", "all zones")
- time: short string describing when (e.g. "late tonight", "Friday morning")
- mechanism: short string describing the cause (e.g. "cold front", "shortwave", "lake enhancement")

Return ONLY a JSON array of objects, no prose. If no events are described, return [].

AFD:
\"\"\"
{text}
\"\"\""""


def _strip_code_fence(s):
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```\s*$", "", s)
    return s.strip()


def call_claude(prompt, timeout=120):
    res = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"claude -p failed: {res.stderr.strip()[:200]}")
    return res.stdout


def extract_frames(text, retries=2):
    prompt = PROMPT.format(text=text)
    last_err = None
    for attempt in range(retries + 1):
        try:
            raw = call_claude(prompt)
            cleaned = _strip_code_fence(raw)
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return parsed
            last_err = f"expected list, got {type(parsed).__name__}"
        except (json.JSONDecodeError, RuntimeError) as e:
            last_err = str(e)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"extract failed after {retries+1} tries: {last_err}")


def _process_one(r):
    out_path = OUT_DIR / f"{r['date']}.json"
    if out_path.exists():
        return r["date"], "skip", None
    try:
        frames = extract_frames(r["text"])
    except Exception as e:
        return r["date"], "fail", str(e)
    out_path.write_text(json.dumps(frames, indent=2))
    return r["date"], "ok", None


def run(limit=None, start_from=None, workers=4):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.PROCESSED_DIR / "paired_dataset.json") as f:
        records = json.load(f)

    if start_from:
        records = [r for r in records if r["date"] >= start_from]
    if limit:
        records = records[:limit]

    done = 0
    failed = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one, r): r for r in records}
        for fut in as_completed(futures):
            date, status, err = fut.result()
            if status == "ok":
                done += 1
            elif status == "fail":
                failed += 1
                print(f"{date} — failed: {err}")
            else:
                skipped += 1
            if (done + failed) % 20 == 0 and (done + failed) > 0:
                print(f"  progress: {done} ok / {failed} failed / {skipped} skip")

    print(f"Done. {done} extracted, {failed} failed, {skipped} skipped. Output in {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="cap on number of AFDs to extract (for testing)")
    parser.add_argument("--start-from", default=None,
                        help="resume from a date (YYYY-MM-DD)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(limit=args.limit, start_from=args.start_from, workers=args.workers)
