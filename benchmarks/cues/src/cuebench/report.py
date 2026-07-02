"""Write results/report.md and results/report.json."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


_RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"


def write(
    sweep_result: Dict[str, Any],
    scan_result: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
) -> tuple[Path, Path]:
    """Serialize results to report.md and report.json.

    Returns (md_path, json_path).
    """
    out = Path(output_dir) if output_dir else _RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "report.json"
    md_path = out / "report.md"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sweep": sweep_result,
        "scan_eval": scan_result,
    }

    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(_render_md(sweep_result, scan_result))

    return md_path, json_path


def _render_md(sweep: Dict[str, Any], scan: Optional[Dict[str, Any]]) -> str:
    lines = [
        "# cuebench report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"Episodes scanned: {sweep.get('episodes_scanned', 0)}",
        f"Floor used: {sweep.get('floor_used', 0.35):.2f}",
        f"Total scores collected: {len(sweep.get('scores', []))}",
        "",
        "## Per-template results",
        "",
    ]

    for tid_str, info in sweep.get("per_template", {}).items():
        label = info.get("label", tid_str)
        cue_type = info.get("cue_type", "")
        dur = info.get("duration_s", 0.0)
        peak = info.get("peak_score", 0.0)
        scores = info.get("scores", [])
        suggestion = info.get("suggestion", {})

        lines.append(f"### Template {tid_str} -- {label} ({cue_type})")
        lines.append("")
        lines.append(f"- Duration: {dur:.3f} s")
        lines.append(f"- Peak score (across all episodes): {peak:.3f}")
        lines.append(f"- Match count (>= floor): {len(scores)}")
        lines.append("")

        # Suggestion
        conf = suggestion.get("confidence", "n/a")
        suggested = suggestion.get("suggested")
        reason = suggestion.get("reason", "")
        if suggested is not None:
            lines.append(
                f"- Suggested threshold: **{suggested}** (confidence: {conf})"
            )
            noise_ceil = suggestion.get("noiseCeiling", "")
            sig_floor = suggestion.get("signalFloor", "")
            gap = suggestion.get("gapWidth", "")
            lines.append(
                f"  - Noise ceiling: {noise_ceil}  Signal floor: {sig_floor}"
                f"  Gap: {gap}"
            )
            warn = suggestion.get("effectFloorWarning")
            if warn:
                lines.append(f"  - Warning: {warn}")
        else:
            lines.append(f"- Suggested threshold: n/a (confidence: {conf})")
            if reason:
                lines.append(f"  - Reason: {reason}")
        lines.append("")

        # Score histogram
        hist = info.get("histogram", {})
        if hist:
            lines.append("**Score histogram (floor 0.35, step 0.05):**")
            lines.append("")
            lines.append("| Bin  | Count |")
            lines.append("|------|-------|")
            for bin_label, count in sorted(hist.items()):
                lines.append(f"| {bin_label} | {count} |")
            lines.append("")

        # Threshold table
        table = info.get("threshold_table", [])
        if table:
            lines.append("**Matches at threshold (0.50-0.95):**")
            lines.append("")
            lines.append("| Threshold | Matches |")
            lines.append("|-----------|---------|")
            for row in table:
                lines.append(
                    f"| {row['threshold']:.2f}      | {row['matches']}       |"
                )
            lines.append("")

    # Formant A/B comparison
    fab = sweep.get("formant_ab")
    if fab:
        lines.append("## Formant A/B comparison (0.0 dB vs 12.0 dB)")
        lines.append("")
        for profile, pt in fab.items():
            lines.append(f"### {profile}")
            for tid_str, info in pt.items():
                label = info.get("label", tid_str)
                lines.append(
                    f"- {label}: {len(info.get('scores', []))} matches,"
                    f" peak {info.get('peak_score', 0.0):.3f}"
                )
        lines.append("")

    # Confirm counts
    confirm = sweep.get("confirm_counts")
    if confirm:
        lines.append("## Confirm counts (re-run at suggested threshold)")
        lines.append("")
        for tid_str, count in confirm.items():
            lines.append(f"- Template {tid_str}: {count} match(es)")
        lines.append("")

    # Scan eval
    if scan:
        lines.append("## Chromaprint discovery eval")
        lines.append("")
        if not scan.get("available"):
            lines.append(f"Skipped: {scan.get('skip_reason', 'unavailable')}")
        else:
            for ep_res in scan.get("results", []):
                lines.append(f"### {ep_res.get('episode', '')}")
                lines.append("")
                for tid_str, info in ep_res.get("per_template", {}).items():
                    label = info.get("label", tid_str)
                    found = info.get("found", False)
                    rank = info.get("rank")
                    span_acc = info.get("span_accuracy")
                    gt = info.get("ground_truth_count", 0)
                    lines.append(
                        f"- {label}: found={found}"
                        f" rank={rank} span_accuracy={span_acc}"
                        f" ground_truth={gt}"
                    )
                lines.append("")

    return "\n".join(lines) + "\n"
