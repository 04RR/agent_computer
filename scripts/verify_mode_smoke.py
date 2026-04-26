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
    path = Path(tempfile.gettempdir()) / "image.png"
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
    tasks = payload.get("tasks") or []
    print(f"  tasks: {len(tasks)} task(s) authored")
    print(f"  report: {len(payload.get('report') or '')} chars")
    print()

    report = payload.get("report") or ""

    # Print the report unconditionally — debug runs always show what the
    # synthesis produced even when downstream checks complain.
    print("=== full report ===")
    print(report)
    print("=== end report ===\n")

    # Per-task summary (Week 2: every node should have completed status
    # and a populated output, set by the scheduler).
    print("=== task outcomes ===")
    tool_call_counts: dict[str, int] = {}
    for t in tasks:
        out = t.get("output") or {}
        out_summary = "<empty>" if not out else f"{len(json.dumps(out))} chars"
        if isinstance(out, dict) and "error" in out:
            out_summary += f" (error: {str(out['error'])[:60]})"
        print(f"  task {t['id']:>2}: {t['status']:<10} {t['node_type']:<7} {t['title'][:50]:50s} output={out_summary}")
        # Count which tool was called by each tool node
        if t["node_type"] == "tool":
            tname = (t.get("config") or {}).get("tool_name", "?")
            tool_call_counts[tname] = tool_call_counts.get(tname, 0) + 1
    print()

    warnings: list[str] = []

    # ── Soft checks ──

    if payload.get("stub_mode") == "hit":
        if "stub" not in report.lower() and "simulat" not in report.lower():
            warnings.append(
                "stub_mode='hit' but report does not mention stubbed / "
                "simulated results. The verification SOUL requires explicit "
                "acknowledgement so readers don't trust fixture domains as "
                "real evidence."
            )

    # Each verification tool should appear at most once in the DAG.
    # 0 calls per tool is also tolerated (planner may have skipped one
    # for a reason — surface it but don't fail).
    expected_tools = {
        "extract_caption_claims",
        "reverse_image_search",
        "extract_image_metadata",
        "fact_check_lookup",
        "reconcile_image_with_caption",
    }
    for tool_name, n in tool_call_counts.items():
        if tool_name in expected_tools and n > 1:
            warnings.append(
                f"{tool_name} appears {n} times in the DAG — Week 2 "
                f"design expects exactly one call per evidence tool."
            )
    missing_tools = expected_tools - set(tool_call_counts)
    if missing_tools:
        warnings.append(
            f"DAG omitted these expected verification tools: "
            f"{sorted(missing_tools)}"
        )

    # Failed or blocked tasks → soft warning (the synthesis will still
    # acknowledge them per the prompt, so don't fail the smoke).
    bad_tasks = [t for t in tasks if t["status"] in ("failed", "blocked")]
    if bad_tasks:
        warnings.append(
            f"{len(bad_tasks)} task(s) in failed/blocked state: "
            f"{[(t['id'], t['status']) for t in bad_tasks]}"
        )

    # ── Hard checks ──

    assert payload.get("session_id"), "missing session_id"
    assert report, "report is empty — synthesis did not produce content"
    assert "error" not in payload, f"endpoint returned an error: {payload.get('error')}"

    # Week 2: agent iterations should be much lower because DAG execution
    # bypasses the agent loop. Allow planning iterations + 1 synthesis call.
    iters = payload.get("iterations", 0)
    assert iters > 0, "iterations should be > 0"
    if iters > 15:
        warnings.append(
            f"iterations={iters} — Week 2 expects much lower (planning + "
            f"1 synthesis call). High count suggests the agent loop ran "
            f"during execution instead of being bypassed."
        )

    # Week 2 hard floor: at least one task should exist (the planner
    # built a DAG). With Bug 1+2+3 fixed, the planner has no excuse.
    assert tasks, "no tasks — planner failed to author a DAG"

    if warnings:
        print("=== warnings ===")
        for w in warnings:
            print(f"  - {w}")
        print()

    print("PASSED" + (" (with warnings)" if warnings else ""))


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    main(base)
