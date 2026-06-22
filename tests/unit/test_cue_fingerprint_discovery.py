"""Unit tests for fingerprint self-repeat cue discovery (#350).

Exercises the pure functions over a synthetic fpcalc ``-raw`` int array: a
distinctive 16-int "sound" planted at several well-separated positions, with the
rest filled by a unique-per-index sequence that never repeats. No fpcalc needed.
"""
from audio_fingerprinter import _discover_repeats, _count_window_matches

FPS = 8.0            # fpcalc -raw emits ~8 ints/sec
FP_DURATION = 100.0  # so n = 800 ints
N = int(FPS * FP_DURATION)
WIN = 16             # AUDIO_CUE_FP_WINDOW_SECONDS (2.0) * FPS


def _noise(i):
    # Knuth multiplicative hash: distinct, well-spread, never-repeating per index.
    return (i * 2654435761) & 0xFFFFFFFF


def _planted_array(positions):
    """An N-int array with a fixed 16-int pattern planted at each position."""
    arr = [_noise(i) for i in range(N)]
    pattern = [(0xA5A5A5A5 ^ (k * 0x01010101)) & 0xFFFFFFFF for k in range(WIN)]
    for p in positions:
        arr[p:p + WIN] = pattern
    return arr


def test_discovers_planted_repeat():
    # Pattern at 80, 320, 560 (>=30s apart, well over the 5s min-gap).
    arr = _planted_array([80, 320, 560])
    cands = _discover_repeats(arr, FP_DURATION, similarity=0.75, min_count=3)
    assert len(cands) == 1
    c = cands[0]
    assert c['count'] == 3
    # ~10.0s; backward extension may walk up to a probe step into the lead-in.
    assert abs(c['start'] - 80 / FPS) <= 1.0


def test_below_min_count_is_dropped():
    # Only two occurrences: below the default min_count of 3.
    arr = _planted_array([80, 320])
    assert _discover_repeats(arr, FP_DURATION, similarity=0.75, min_count=3) == []


def test_unique_noise_yields_no_candidates():
    arr = [_noise(i) for i in range(N)]
    assert _discover_repeats(arr, FP_DURATION, similarity=0.75, min_count=3) == []


def test_empty_or_degenerate_input():
    assert _discover_repeats([], FP_DURATION, similarity=0.75, min_count=3) == []
    assert _discover_repeats([1, 2, 3], 0.0, similarity=0.75, min_count=3) == []


def test_count_self_matches_recurring_window():
    arr = _planted_array([80, 320, 560])
    # The planted window [10s, 12s] recurs three times.
    assert _count_window_matches(arr, FP_DURATION, 10.0, 12.0, similarity=0.75) == 3


def test_count_self_matches_one_off_window():
    arr = _planted_array([80])
    # Window over a non-recurring noise stretch appears only once.
    assert _count_window_matches(arr, FP_DURATION, 50.0, 52.0, similarity=0.75) == 1
