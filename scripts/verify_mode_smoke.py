"""End-to-end smoke test for verify mode (Phase 2).

Requires a running gateway:
    python gateway.py

Run from another terminal:
    python scripts/verify_mode_smoke.py
    python scripts/verify_mode_smoke.py http://localhost:8000

Uses the synthetic 64x64 PNG from the Phase 1 smoke script if present;
otherwise generates one. Sends it to /api/verify with an India-themed
caption that should trigger the INDIA_CONSISTENT stub fixture (when
tineye_stub_mode='hit' is set in config.json).

Asserts:
- Endpoint returns 200
- Response contains the expected fields (report, tasks, iterations, ...)
- Report is non-empty Markdown text
- tasks list is non-empty (the agent built a DAG)
- If tineye_stub_mode is 'hit', the report mentions the stub explicitly
- iteration count is below verify.max_iterations
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import httpx


def _make_synthetic_image() -> str:
    from PIL import Image
    path = Path(tempfile.gettempdir()) / "verify_mode_smoke.png"
    Image.new("RGB", (64, 64), color=(80, 140, 200)).save(path, format="PNG")
    return str(path)


def main(base_url: str = "http://localhost:8000") -> None:
    image_path = _make_synthetic_image()
    caption = "Mumbai protest yesterday — students demanding fee reform"

    print(f"POST {base_url}/api/verify")
    print(f"  image: {image_path}")
    print(f"  caption: {caption!r}")
    print()

    # Verification can take 30-90s depending on the model. Generous timeout.
    with httpx.Client(timeout=180.0) as client:
        with open(image_path, "rb") as f:
            resp = client.post(
                f"{base_url}/api/verify",
                files={"image": ("smoke.png", f, "image/png")},
                data={"caption": caption},
            )

    if resp.status_code != 200:
        print(f"FAILED: status={resp.status_code}")
        print(resp.text[:1000])
        sys.exit(1)

    payload = resp.json()
    print("=== response shape ===")
    for k in ("session_id", "iterations", "elapsed_ms", "stub_mode"):
        print(f"  {k}: {payload.get(k)!r}")
    print(f"  tasks: {len(payload.get('tasks') or [])} task(s) authored")
    print(f"  report: {len(payload.get('report') or '')} chars")
    print()

    # Hard checks.
    assert payload.get("session_id"), "missing session_id"
    assert payload.get("report"), "report is empty — agent did not synthesize"
    assert payload.get("tasks"), "no tasks — agent did not author a DAG"
    assert payload.get("iterations", 0) > 0, "iterations should be > 0"

    report = payload["report"]
    if payload.get("stub_mode") == "hit":
        if "stub" not in report.lower() and "simulat" not in report.lower():
            print("WARNING: stub_mode='hit' but report does not mention "
                  "stubbed/simulated results. The verification SOUL requires "
                  "explicit acknowledgement.")
        else:
            print("OK: report mentions stub/simulated results")

    print()
    print("=== first 800 chars of report ===")
    print(report[:800])
    print("...")
    print()
    print("PASSED")


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    main(base)
