"""Evaluate Chromaprint-based recurring-sound discovery against sweep ground truth."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger("cuebench.scan_eval")

from config import (
    AUDIO_CUE_RECURRENCE_SIMILARITY,
    AUDIO_CUE_RECURRENCE_MIN_COUNT,
)


def run(
    audio_paths: List[Path],
    sweep_per_template: Dict[str, Any],
    template_durations: Dict[str, float],
) -> Dict[str, Any]:
    """Run AudioFingerprinter discovery on *audio_paths* and compare with sweep.

    Skips cleanly (returns skip_reason) when fpcalc is unavailable.

    Ground truth: positions from the sweep (score >= floor) are treated as
    known occurrences for found?/rank/span accuracy evaluation.

    Returns a dict:
      available: bool
      skip_reason: str | None
      results: list[dict]  -- one per episode
    """
    try:
        from audio_fingerprinter import AudioFingerprinter
    except ImportError as e:
        return {
            "available": False,
            "skip_reason": f"could not import AudioFingerprinter: {e}",
            "results": [],
        }

    fp = AudioFingerprinter(db=None)
    if not fp.is_available():
        if shutil.which("fpcalc") is None:
            reason = "fpcalc not found on PATH"
        else:
            reason = "AudioFingerprinter reports unavailable"
        return {"available": False, "skip_reason": reason, "results": []}

    episode_results = []
    for path in audio_paths:
        candidates = fp.discover_recurring_spots(
            str(path),
            similarity=AUDIO_CUE_RECURRENCE_SIMILARITY,
            min_count=AUDIO_CUE_RECURRENCE_MIN_COUNT,
        )
        episode_results.append(
            _eval_episode(path, candidates, sweep_per_template, template_durations)
        )

    return {"available": True, "skip_reason": None, "results": episode_results}


def _eval_episode(
    path: Path,
    candidates: List[Dict],
    sweep_per_template: Dict[str, Any],
    template_durations: Dict[str, float],
) -> Dict[str, Any]:
    per_template = {}
    for tid_str, info in sweep_per_template.items():
        gt_scores = info.get("scores", [])
        tpl_dur = template_durations.get(tid_str, info.get("duration_s", 0.0))
        label = info.get("label", tid_str)
        ground_truth_count = len(gt_scores)

        found = False
        rank: Optional[int] = None
        span_accuracy: Optional[float] = None

        for i, cand in enumerate(candidates):
            start = cand.get("start", 0.0)
            end = cand.get("end", 0.0)
            span = end - start
            # Consider a candidate a match if its span is within 50% of the
            # template duration -- coarse but sufficient for discovery eval.
            if tpl_dur > 0 and abs(span - tpl_dur) / tpl_dur <= 0.5:
                found = True
                rank = i + 1
                span_accuracy = round(1.0 - abs(span - tpl_dur) / max(tpl_dur, 1e-6), 3)
                break

        per_template[tid_str] = {
            "label": label,
            "ground_truth_count": ground_truth_count,
            "found": found,
            "rank": rank,
            "span_accuracy": span_accuracy,
            "candidates_total": len(candidates),
        }

    return {"episode": str(path), "per_template": per_template}
