"""Tests for cue-candidate merging (recurring + one-off loud spots).

The candidate scan used to surface only sounds that recur >= 3x in an episode,
so a once-per-episode intro/outro/bumper never appeared. merge_cue_candidates
folds in one-off loud spots and tags every candidate with a kind and a
positional cue-type hint so candidates cover all cue types.
"""
from audio_analysis.cue_candidates import (
    merge_cue_candidates, candidate_suggested_type,
)
from config import (
    AUDIO_CUE_CANDIDATE_MAX_RESULTS,
    AUDIO_CUE_TYPE_SHOW_INTRO,
    AUDIO_CUE_TYPE_SHOW_OUTRO,
)


class TestSuggestedType:
    def test_near_start_is_intro(self):
        assert candidate_suggested_type(3.0, 7.0, 1800.0) == AUDIO_CUE_TYPE_SHOW_INTRO

    def test_near_end_is_outro(self):
        assert candidate_suggested_type(1790.0, 1799.0, 1800.0) == AUDIO_CUE_TYPE_SHOW_OUTRO

    def test_middle_is_untyped(self):
        assert candidate_suggested_type(900.0, 905.0, 1800.0) is None

    def test_unknown_duration_skips_outro_but_still_types_intro(self):
        assert candidate_suggested_type(1790.0, 1799.0, None) is None
        assert candidate_suggested_type(3.0, 7.0, None) == AUDIO_CUE_TYPE_SHOW_INTRO


class TestMergeCueCandidates:
    def test_recurring_tagged_with_kind_and_count(self):
        out = merge_cue_candidates([{'start': 400.0, 'end': 404.0, 'count': 4}], [], 1800.0)
        assert out == [{
            'start': 400.0, 'end': 404.0, 'kind': 'recurring',
            'count': 4, 'suggestedType': None,
        }]

    def test_loud_spot_overlapping_a_recurring_hit_is_dropped(self):
        recurring = [{'start': 400.0, 'end': 410.0, 'count': 3}]
        loud = [{'start': 405.0, 'end': 408.0, 'prominenceDb': 20.0}]
        out = merge_cue_candidates(recurring, loud, 1800.0)
        assert len(out) == 1
        assert out[0]['kind'] == 'recurring'

    def test_non_overlapping_loud_spot_becomes_one_off(self):
        loud = [{'start': 900.0, 'end': 904.0, 'prominenceDb': 12.0}]
        out = merge_cue_candidates([], loud, 1800.0)
        assert out == [{
            'start': 900.0, 'end': 904.0, 'kind': 'one_off',
            'prominenceDb': 12.0, 'suggestedType': None,
        }]

    def test_recurring_ranks_before_one_off(self):
        recurring = [{'start': 900.0, 'end': 903.0, 'count': 3}]
        loud = [{'start': 200.0, 'end': 205.0, 'prominenceDb': 99.0}]
        out = merge_cue_candidates(recurring, loud, 1800.0)
        assert [c['kind'] for c in out] == ['recurring', 'one_off']

    def test_recurring_sorted_by_count_desc(self):
        recurring = [
            {'start': 500.0, 'end': 503.0, 'count': 2},
            {'start': 900.0, 'end': 903.0, 'count': 7},
        ]
        out = merge_cue_candidates(recurring, [], 1800.0)
        assert [c['count'] for c in out] == [7, 2]

    def test_one_offs_sorted_by_prominence_desc(self):
        loud = [
            {'start': 500.0, 'end': 503.0, 'prominenceDb': 6.0},
            {'start': 900.0, 'end': 903.0, 'prominenceDb': 18.0},
        ]
        out = merge_cue_candidates([], loud, 1800.0)
        assert [c['prominenceDb'] for c in out] == [18.0, 6.0]

    def test_positional_typing_applied_to_both_kinds(self):
        recurring = [{'start': 2.0, 'end': 6.0, 'count': 3}]      # intro
        loud = [{'start': 1795.0, 'end': 1799.0, 'prominenceDb': 9.0}]  # outro
        out = merge_cue_candidates(recurring, loud, 1800.0)
        by_kind = {c['kind']: c['suggestedType'] for c in out}
        assert by_kind['recurring'] == AUDIO_CUE_TYPE_SHOW_INTRO
        assert by_kind['one_off'] == AUDIO_CUE_TYPE_SHOW_OUTRO

    def test_result_is_capped(self):
        recurring = [
            {'start': float(i * 10), 'end': float(i * 10 + 2), 'count': 3}
            for i in range(AUDIO_CUE_CANDIDATE_MAX_RESULTS + 5)
        ]
        out = merge_cue_candidates(recurring, [], 1800.0)
        assert len(out) == AUDIO_CUE_CANDIDATE_MAX_RESULTS
