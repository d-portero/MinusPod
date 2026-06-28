"""Pure merging of cue-template candidates (recurring + one-off loud spots).

Kept free of Flask/DB imports so the candidate-scan worker and unit tests can
use it directly. The worker (api/cue_templates.py) supplies the two candidate
lists; this module tags, dedupes, and ranks them.
"""
from config import (
    AUDIO_CUE_INTRO_WINDOW_SECONDS, AUDIO_CUE_OUTRO_WINDOW_SECONDS,
    AUDIO_CUE_CANDIDATE_MAX_RESULTS,
    AUDIO_CUE_TYPE_SHOW_INTRO, AUDIO_CUE_TYPE_SHOW_OUTRO,
)
from utils.time import ranges_overlap


def candidate_suggested_type(start, end, episode_duration):
    """Positional cue-type hint: intro near the episode start, outro near the
    end, else none. A default the capture UI can override."""
    if start <= AUDIO_CUE_INTRO_WINDOW_SECONDS:
        return AUDIO_CUE_TYPE_SHOW_INTRO
    if episode_duration and end >= episode_duration - AUDIO_CUE_OUTRO_WINDOW_SECONDS:
        return AUDIO_CUE_TYPE_SHOW_OUTRO
    return None


def _sort_key(c):
    """Recurring first (most-recurring), then one-offs (most prominent)."""
    if c['kind'] == 'recurring':
        return (0, -(c.get('count') or 0))
    return (1, -(c.get('prominenceDb') or 0.0))


def merge_cue_candidates(recurring, loud_spots, episode_duration):
    """Merge recurrence hits and one-off loud spots into typed candidates.

    A real cue may recur (caught by the fingerprint scan) OR play once -- a show
    intro/outro/bumper often plays a single time per episode, which the
    recurrence scan never surfaces. Loud spots fill that gap so candidates cover
    all cue types, not just repeated sounds. A loud spot overlapping a recurrence
    hit is dropped (the recurrence hit is the stronger signal). Each candidate is
    tagged with its kind and a positional cue-type hint; recurring candidates
    rank first (by recurrence count), then one-offs (by prominence).
    """
    merged = [
        {
            'start': c['start'], 'end': c['end'], 'kind': 'recurring',
            'count': c.get('count'),
            'suggestedType': candidate_suggested_type(
                c['start'], c['end'], episode_duration),
        }
        for c in recurring
    ]
    for spot in loud_spots:
        if any(ranges_overlap(spot['start'], spot['end'], r['start'], r['end'])
               for r in recurring):
            continue
        merged.append({
            'start': spot['start'], 'end': spot['end'], 'kind': 'one_off',
            'prominenceDb': spot.get('prominenceDb'),
            'suggestedType': candidate_suggested_type(
                spot['start'], spot['end'], episode_duration),
        })
    merged.sort(key=_sort_key)
    return merged[:AUDIO_CUE_CANDIDATE_MAX_RESULTS]
