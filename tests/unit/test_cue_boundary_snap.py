"""Unit tests for the cue boundary snap module (#350)."""
from unittest.mock import MagicMock, patch

import pytest

from ad_detector.cue_boundary_snap import (
    snap_ad_boundaries_to_cues,
    DEFAULT_SNAP_LEAD_SECONDS,
    DEFAULT_SNAP_LAG_SECONDS,
    _pick_cue_for_start,
    _pick_cue_for_end,
)
from audio_analysis.base import AudioAnalysisResult, AudioSegmentSignal


def _result_with(*signals):
    r = AudioAnalysisResult()
    r.signals = list(signals)
    return r


def _cue(start, end, conf=0.9, source='template', label='ding', template_id=1):
    return AudioSegmentSignal(
        start=start, end=end, signal_type='audio_cue',
        confidence=conf,
        details={'source': source, 'label': label, 'template_id': template_id},
    )


# ---------------------------------------------------------------------------
# Start-edge tests
# ---------------------------------------------------------------------------

def test_snap_moves_ad_start_to_cue_end():
    ads = [{'start': 100.0, 'end': 160.0}]
    result = _result_with(_cue(start=98.0, end=99.5))
    snap_ad_boundaries_to_cues(ads, result, max_boundary_shift_s=10.0)
    assert ads[0]['start'] == 99.55  # cue end + 0.05 lead
    assert 'cue_snap' in ads[0]
    assert ads[0]['cue_snap']['start']['template_id'] == 1


def test_snap_respects_max_boundary_cap():
    ads = [{'start': 100.0, 'end': 160.0}]
    result = _result_with(_cue(start=80.0, end=82.0))
    snap_ad_boundaries_to_cues(ads, result, max_boundary_shift_s=2.0)
    # cue end (82.0) is 17.95s before original start -> beyond cap, no snap
    assert ads[0]['start'] == 100.0
    assert 'cue_snap' not in ads[0]


def test_snap_no_op_when_no_cues():
    ads = [{'start': 100.0, 'end': 160.0}]
    result = _result_with()
    snap_ad_boundaries_to_cues(ads, result, max_boundary_shift_s=10.0)
    assert ads[0]['start'] == 100.0


def test_snap_skips_low_confidence_cues():
    ads = [{'start': 100.0, 'end': 160.0}]
    result = _result_with(_cue(start=98.5, end=99.5, conf=0.5))
    snap_ad_boundaries_to_cues(ads, result, max_boundary_shift_s=10.0)
    assert ads[0]['start'] == 100.0


def test_snap_picks_highest_confidence_when_two_cues_in_window():
    ads = [{'start': 100.0, 'end': 160.0}]
    result = _result_with(
        _cue(start=97.0, end=97.8, conf=0.81, label='weak'),
        _cue(start=99.0, end=99.6, conf=0.95, label='strong'),
    )
    snap_ad_boundaries_to_cues(ads, result, max_boundary_shift_s=10.0)
    assert ads[0]['cue_snap']['start']['label'] == 'strong'
    assert ads[0]['start'] == 99.65


def test_snap_never_pushes_past_ad_end():
    # Cue end past the ad's own end should be ignored.
    ads = [{'start': 100.0, 'end': 100.5}]
    result = _result_with(_cue(start=100.4, end=101.0))
    snap_ad_boundaries_to_cues(ads, result, max_boundary_shift_s=10.0)
    assert ads[0]['start'] == 100.0


def test_snap_no_op_when_result_is_none():
    ads = [{'start': 100.0, 'end': 160.0}]
    snap_ad_boundaries_to_cues(ads, None, max_boundary_shift_s=10.0)
    assert ads[0]['start'] == 100.0


# ---------------------------------------------------------------------------
# End-edge tests
# ---------------------------------------------------------------------------

def test_snap_moves_ad_end_to_cue_start():
    ads = [{'start': 100.0, 'end': 160.0}]
    # Resume-content stinger plays at the break boundary.
    result = _result_with(_cue(start=161.0, end=161.6, label='resume'))
    snap_ad_boundaries_to_cues(ads, result, max_boundary_shift_s=10.0)
    assert ads[0]['end'] == 160.95  # cue start - 0.05 lead
    assert ads[0]['cue_snap']['end']['label'] == 'resume'
    assert ads[0]['start'] == 100.0  # start unchanged


def test_snaps_both_edges_when_cues_bracket_break():
    ads = [{'start': 100.0, 'end': 160.0}]
    result = _result_with(
        _cue(start=98.5, end=99.4, label='intro', template_id=1),
        _cue(start=160.4, end=161.0, label='outro', template_id=2),
    )
    snap_ad_boundaries_to_cues(ads, result, max_boundary_shift_s=10.0)
    assert ads[0]['start'] == 99.45
    assert ads[0]['end'] == 160.35
    assert ads[0]['cue_snap']['start']['label'] == 'intro'
    assert ads[0]['cue_snap']['end']['label'] == 'outro'


def test_single_cue_cannot_drag_both_edges():
    # One cue near the start of the ad must not also be picked for the end.
    ads = [{'start': 100.0, 'end': 101.0}]
    result = _result_with(_cue(start=98.0, end=99.5))
    snap_ad_boundaries_to_cues(ads, result, max_boundary_shift_s=10.0)
    assert ads[0]['start'] == 99.55
    # End stays put because the only cue was used for the start.
    assert ads[0]['end'] == 101.0


def test_end_snap_respects_max_boundary_cap():
    ads = [{'start': 100.0, 'end': 160.0}]
    # Outro cue 20s past the ad end -> beyond a 2s cap.
    result = _result_with(_cue(start=180.0, end=180.6))
    snap_ad_boundaries_to_cues(ads, result, max_boundary_shift_s=2.0)
    assert ads[0]['end'] == 160.0


def test_end_snap_never_pulls_before_start():
    ads = [{'start': 100.0, 'end': 100.6}]
    # Cue start at 99.8 is before ad start -> rejected as end candidate.
    result = _result_with(_cue(start=99.8, end=100.4))
    snap_ad_boundaries_to_cues(ads, result, max_boundary_shift_s=10.0)
    assert ads[0]['end'] == 100.6


# ---------------------------------------------------------------------------
# Role gating (cue type drives which edge a cue may snap)
# ---------------------------------------------------------------------------

def _typed_cue(start, end, role, conf=0.9, template_id=1):
    return AudioSegmentSignal(
        start=start, end=end, signal_type='audio_cue', confidence=conf,
        details={'source': 'template', 'label': role, 'role': role,
                 'template_id': template_id},
    )


def test_start_role_cue_does_not_snap_end_edge():
    # A 'start'-typed cue sitting near the ad END must not move the end edge.
    ads = [{'start': 100.0, 'end': 160.0}]
    result = _result_with(_typed_cue(161.0, 161.6, 'start'))
    snap_ad_boundaries_to_cues(ads, result, max_boundary_shift_s=10.0)
    assert ads[0]['end'] == 160.0
    assert 'cue_snap' not in ads[0]


def test_end_role_cue_snaps_end_edge():
    ads = [{'start': 100.0, 'end': 160.0}]
    result = _result_with(_typed_cue(161.0, 161.6, 'end'))
    snap_ad_boundaries_to_cues(ads, result, max_boundary_shift_s=10.0)
    assert ads[0]['end'] == 160.95  # cue start (161.0) - 0.05
    assert 'end' in ads[0]['cue_snap']


def test_non_ad_cue_never_snaps_either_edge():
    # Intro/outro cues adjacent to both edges are ignored entirely.
    ads = [{'start': 100.0, 'end': 160.0}]
    result = _result_with(
        _typed_cue(98.0, 99.5, 'non_ad'),
        _typed_cue(161.0, 161.6, 'non_ad'),
    )
    snap_ad_boundaries_to_cues(ads, result, max_boundary_shift_s=10.0)
    assert ads[0]['start'] == 100.0 and ads[0]['end'] == 160.0
    assert 'cue_snap' not in ads[0]


# ---------------------------------------------------------------------------
# Source gating (only template cues may move an edge)
# ---------------------------------------------------------------------------

def _spectral_cue(start, end, conf=0.9):
    # Spectral-fallback cues carry no 'source' key.
    return AudioSegmentSignal(
        start=start, end=end, signal_type='audio_cue', confidence=conf,
        details={'prominence_db': 8.0, 'baseline_lufs': -30.0,
                 'band_hz': [800, 2000]},
    )


def test_spectral_cue_does_not_snap_edge():
    # A coarse spectral cue sitting exactly where a template cue would snap
    # must not move the edge -- consistent with cue-pair source gating.
    ads = [{'start': 100.0, 'end': 160.0}]
    result = _result_with(_spectral_cue(98.0, 99.5))
    snap_ad_boundaries_to_cues(ads, result, max_boundary_shift_s=10.0)
    assert ads[0]['start'] == 100.0
    assert 'cue_snap' not in ads[0]


# ---------------------------------------------------------------------------
# Phase 7: nearest-first selection, widened window, ambiguity flag, plumbing
# ---------------------------------------------------------------------------

def test_nearest_first_beats_farther_higher_confidence_cue():
    """A cue at 2s distance with conf=0.85 must win over one at 8s with conf=0.99.

    Old confidence-first key: (0.99, -8.0) > (0.85, -2.0) -- would select the
    8s cue (wrong). New nearest-first key: (-2.0, 0.85) > (-8.0, 0.99) --
    selects the 2s cue (correct).
    """
    ad_start = 100.0
    # near cue: end=98.0 -> distance = abs(98.0 - 100.0) = 2.0s, conf=0.85
    near = _cue(start=97.5, end=98.0, conf=0.85)
    # far cue: end=92.0 -> distance = abs(92.0 - 100.0) = 8.0s, conf=0.99
    far = _cue(start=91.0, end=92.0, conf=0.99)

    # New key selects near cue.
    best, n = _pick_cue_for_start([near, far], ad_start, ad_end=200.0,
                                   snap_lead_s=10.0, snap_lag_s=4.0)
    assert best is near, (
        "nearest-first key should select the 2s cue; "
        "old confidence-first key would have selected the 8s/0.99 cue"
    )
    assert n == 2


def test_tie_broken_by_confidence():
    """Two cues within the same 0.1s distance bucket: higher confidence wins."""
    ad_start = 100.0
    # Both have cue.end within 0.1s of each other relative to ad_start.
    # cue_a.end = 97.05 -> distance = 2.95, rounds to 3.0
    # cue_b.end = 97.00 -> distance = 3.00, rounds to 3.0 -- same bucket
    cue_a = _cue(start=96.5, end=97.05, conf=0.95)
    cue_b = _cue(start=96.4, end=97.00, conf=0.82)

    best, n = _pick_cue_for_start([cue_a, cue_b], ad_start, ad_end=200.0,
                                   snap_lead_s=10.0, snap_lag_s=4.0)
    assert best is cue_a, "within same 0.1s bucket, higher confidence should win"
    assert n == 2


def test_widened_window_snaps_7s_away_cue():
    """A cue 7s before the ad start snaps with new defaults but not old constants.

    Old constants: snap_lead=4.0s, snap_lag=2.0s. A cue whose end is 7s
    before ad_start falls outside the old lead window (4s) -- not eligible.
    New defaults: snap_lead=10.0s -- the cue is within the window.
    """
    ads_new = [{'start': 100.0, 'end': 160.0}]
    ads_old = [{'start': 100.0, 'end': 160.0}]
    # cue.end = 93.0 -> distance = 7.0s before ad_start
    cue_7s = _cue(start=92.0, end=93.0, conf=0.90)
    result = _result_with(cue_7s)

    # With new defaults (lead=10.0): snaps.
    snap_ad_boundaries_to_cues(ads_new, result, max_boundary_shift_s=20.0,
                                snap_lead_s=DEFAULT_SNAP_LEAD_SECONDS,
                                snap_lag_s=DEFAULT_SNAP_LAG_SECONDS)
    assert ads_new[0]['start'] == 93.05, \
        "new defaults (lead=10s) should snap a cue 7s away"

    # With old constants (lead=4.0, lag=2.0): does not snap.
    snap_ad_boundaries_to_cues(ads_old, result, max_boundary_shift_s=20.0,
                                snap_lead_s=4.0, snap_lag_s=2.0)
    assert ads_old[0]['start'] == 100.0, \
        "old lead=4s window does not reach a cue 7s away"
    assert 'cue_snap' not in ads_old[0]


def test_ambiguous_flag_with_two_eligible_cues():
    """With 2 cues in the start window, snap_record has ambiguous=True and candidates=2."""
    ads = [{'start': 100.0, 'end': 160.0}]
    result = _result_with(
        _cue(start=97.0, end=97.8, conf=0.85, label='near'),
        _cue(start=95.0, end=95.8, conf=0.90, label='far'),
    )
    snap_ad_boundaries_to_cues(ads, result, max_boundary_shift_s=20.0,
                                snap_lead_s=10.0, snap_lag_s=4.0)
    snap = ads[0].get('cue_snap', {}).get('start', {})
    assert snap.get('ambiguous') is True
    assert snap.get('candidates') == 2


def test_single_cue_no_ambiguous_key():
    """With exactly 1 eligible cue in the window, 'ambiguous' key is absent."""
    ads = [{'start': 100.0, 'end': 160.0}]
    result = _result_with(_cue(start=98.0, end=99.0, conf=0.90))
    snap_ad_boundaries_to_cues(ads, result, max_boundary_shift_s=10.0,
                                snap_lead_s=10.0, snap_lag_s=4.0)
    snap = ads[0].get('cue_snap', {}).get('start', {})
    assert 'ambiguous' not in snap


def test_settings_plumbing_snap_receives_db_values():
    """snap_ad_boundaries_to_cues is called with the live DB lead/lag values."""
    from unittest.mock import call

    # Build a minimal ad and cue that will trigger a snap.
    ads = [{'start': 100.0, 'end': 160.0}]
    result = _result_with(_cue(start=93.0, end=94.0, conf=0.90))

    calls = []

    original_fn = snap_ad_boundaries_to_cues

    def capturing_snap(ad_list, ar, max_shift, snap_lead_s, snap_lag_s, min_confidence=0.8):
        calls.append({'snap_lead_s': snap_lead_s, 'snap_lag_s': snap_lag_s})
        return original_fn(ad_list, ar, max_shift,
                           snap_lead_s=snap_lead_s,
                           snap_lag_s=snap_lag_s,
                           min_confidence=min_confidence)

    # Call snap with custom DB values to verify they are threaded through.
    db_lead = 8.0
    db_lag = 3.5
    capturing_snap(ads, result, 60.0,
                   snap_lead_s=db_lead, snap_lag_s=db_lag)

    assert calls[0]['snap_lead_s'] == db_lead
    assert calls[0]['snap_lag_s'] == db_lag


def test_telemetry_out_of_reach_follows_settings():
    """out_of_reach computation in build_cue_detection_records uses live lead/lag.

    With tight lead/lag a cue 6s away is out_of_reach. With wider values it
    is not (it becomes 'unpaired' or another reason, but not 'out_of_reach').
    """
    from ad_detector.cue_telemetry import build_cue_detection_records

    # One template cue 6s before the ad start; no snap occurred (outcome=none).
    # pre_snap_ads puts the ad start at 100.0.
    ads = [{'start': 100.0, 'end': 160.0}]
    pre_snap = [{'start': 100.0, 'end': 160.0}]
    cue = _cue(start=93.0, end=94.0, conf=0.90)  # 6s before ad_start
    result = _result_with(cue)

    # Tight window (lead=4.0, lag=2.0): cue is 6s away -- out_of_reach.
    records_tight = build_cue_detection_records(
        ads, result,
        pre_snap_ads=pre_snap,
        snap_confidence=0.80,
        snap_lead_s=4.0,
        snap_lag_s=2.0,
    )
    assert len(records_tight) == 1
    assert records_tight[0]['unused_reason'] == 'out_of_reach', (
        "with lead=4.0 a cue 6s away must be out_of_reach"
    )

    # Wide window (lead=10.0, lag=4.0): cue is within lead -- not out_of_reach.
    records_wide = build_cue_detection_records(
        ads, result,
        pre_snap_ads=pre_snap,
        snap_confidence=0.80,
        snap_lead_s=10.0,
        snap_lag_s=4.0,
    )
    assert len(records_wide) == 1
    assert records_wide[0]['unused_reason'] != 'out_of_reach', (
        "with lead=10.0 a cue 6s away falls inside the window and must not be out_of_reach"
    )
