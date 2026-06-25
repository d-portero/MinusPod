"""Unit tests for the recut cut-list helpers (issue #422)."""
import json
import os
import sys
import tempfile

# Boot pattern (see test_history_ad_count.py): bind a temp DATA_DIR before
# importing main_app, which otherwise tries to mkdir /app/data at module-load.
_test_data_dir = tempfile.mkdtemp(prefix='recut_test_')
os.environ.setdefault('SECRET_KEY', 'test-secret')
os.environ['DATA_DIR'] = _test_data_dir

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import database
import storage as storage_mod

database.Database._instance = None
database.Database.__init__.__defaults__ = (_test_data_dir,)
database.Database.__new__.__defaults__ = (_test_data_dir,)
storage_mod.Storage.__init__.__defaults__ = (_test_data_dir,)

from main_app import processing


def test_best_overlap_ad_picks_max_overlap():
    ads = [{'start': 0, 'end': 10}, {'start': 50, 'end': 70}, {'start': 100, 'end': 110}]
    assert processing._best_overlap_ad(ads, 55, 65) is ads[1]


def test_best_overlap_ad_none_when_no_overlap():
    ads = [{'start': 0, 'end': 10}]
    assert processing._best_overlap_ad(ads, 50, 60) is None


def test_best_overlap_ad_excludes_ids():
    ads = [{'start': 0, 'end': 10}, {'start': 0, 'end': 10}]
    first = processing._best_overlap_ad(ads, 1, 5)
    second = processing._best_overlap_ad(ads, 1, 5, exclude_ids={id(first)})
    assert second is not first


def test_apply_boundary_adjustments_overrides_bounds(monkeypatch):
    ads = [{'start': 100.0, 'end': 160.0, 'confidence': 0.9}]
    corrections = [{
        'correction_type': 'boundary_adjustment',
        'original_bounds': {'start': 100.0, 'end': 160.0},
        'corrected_bounds': {'start': 105.0, 'end': 150.0},
    }]
    monkeypatch.setattr(processing.db, 'get_episode_corrections', lambda eid: corrections)
    processing._apply_boundary_adjustments('slug', 'ep', ads)
    assert ads[0]['start'] == 105.0
    assert ads[0]['end'] == 150.0


def test_apply_boundary_adjustments_skips_unmatched(monkeypatch):
    ads = [{'start': 100.0, 'end': 160.0}]
    corrections = [{
        'correction_type': 'boundary_adjustment',
        'original_bounds': {'start': 900.0, 'end': 950.0},
        'corrected_bounds': {'start': 905.0, 'end': 940.0},
    }]
    monkeypatch.setattr(processing.db, 'get_episode_corrections', lambda eid: corrections)
    processing._apply_boundary_adjustments('slug', 'ep', ads)
    assert ads[0]['start'] == 100.0
    assert ads[0]['end'] == 160.0


def test_apply_boundary_adjustments_newest_wins(monkeypatch):
    ads = [{'start': 100.0, 'end': 160.0}]
    # get_episode_corrections returns newest first (ORDER BY created_at DESC).
    corrections = [
        {'correction_type': 'boundary_adjustment',
         'original_bounds': {'start': 100.0, 'end': 160.0},
         'corrected_bounds': {'start': 110.0, 'end': 150.0}},
        {'correction_type': 'boundary_adjustment',
         'original_bounds': {'start': 100.0, 'end': 160.0},
         'corrected_bounds': {'start': 101.0, 'end': 159.0}},
    ]
    monkeypatch.setattr(processing.db, 'get_episode_corrections', lambda eid: corrections)
    processing._apply_boundary_adjustments('slug', 'ep', ads)
    assert ads[0]['start'] == 110.0
    assert ads[0]['end'] == 150.0


def test_build_recut_ad_list_drops_rejected(monkeypatch):
    ads = [
        {'start': 30.0, 'end': 90.0, 'confidence': 0.98, 'sponsor': 'A', 'reason': 'sponsor read for A'},
        {'start': 300.0, 'end': 360.0, 'confidence': 0.98, 'sponsor': 'B', 'reason': 'sponsor read for B'},
    ]
    monkeypatch.setattr(processing.db, 'get_episode', lambda s, e: {'ad_markers_json': json.dumps(ads)})
    monkeypatch.setattr(processing.db, 'get_episode_corrections', lambda eid: [])
    monkeypatch.setattr(processing.db, 'get_false_positive_corrections',
                        lambda eid: [{'start': 300.0, 'end': 360.0}])
    monkeypatch.setattr(processing.db, 'get_confirmed_corrections', lambda eid: [])
    segments = [
        {'start': 30.0, 'end': 90.0, 'text': 'sponsor a'},
        {'start': 300.0, 'end': 360.0, 'text': 'sponsor b'},
    ]
    ads_to_remove, all_ads = processing._build_recut_ad_list(
        'slug', 'ep', segments, 600.0, '', 0.80
    )
    starts = {a['start'] for a in ads_to_remove}
    assert 30.0 in starts
    assert 300.0 not in starts
    assert len(all_ads) == 2  # rejected ad stays in the list, just not cut


def test_build_recut_ad_list_keeps_confirmed(monkeypatch):
    # A low-confidence ad the user confirmed must still be cut.
    ads = [{'start': 30.0, 'end': 90.0, 'confidence': 0.40, 'sponsor': 'A', 'reason': 'maybe an ad for A'}]
    monkeypatch.setattr(processing.db, 'get_episode', lambda s, e: {'ad_markers_json': json.dumps(ads)})
    monkeypatch.setattr(processing.db, 'get_episode_corrections', lambda eid: [])
    monkeypatch.setattr(processing.db, 'get_false_positive_corrections', lambda eid: [])
    monkeypatch.setattr(processing.db, 'get_confirmed_corrections', lambda eid: [{'start': 30.0, 'end': 90.0}])
    segments = [{'start': 30.0, 'end': 90.0, 'text': 'maybe an ad'}]
    ads_to_remove, _ = processing._build_recut_ad_list('slug', 'ep', segments, 600.0, '', 0.80)
    assert {a['start'] for a in ads_to_remove} == {30.0}


def test_build_recut_ad_list_keeps_manual_add(monkeypatch):
    # A manually-added marker (confidence 1.0, detection_stage 'manual') must be cut.
    ads = [{'start': 120.0, 'end': 180.0, 'confidence': 1.0, 'detection_stage': 'manual',
            'sponsor': 'Manual Co', 'reason': 'Manual Co: manually added ad'}]
    monkeypatch.setattr(processing.db, 'get_episode', lambda s, e: {'ad_markers_json': json.dumps(ads)})
    monkeypatch.setattr(processing.db, 'get_episode_corrections', lambda eid: [])
    monkeypatch.setattr(processing.db, 'get_false_positive_corrections', lambda eid: [])
    monkeypatch.setattr(processing.db, 'get_confirmed_corrections', lambda eid: [])
    segments = [{'start': 120.0, 'end': 180.0, 'text': 'manual'}]
    ads_to_remove, _ = processing._build_recut_ad_list('slug', 'ep', segments, 600.0, '', 0.80)
    assert {a['start'] for a in ads_to_remove} == {120.0}


def test_build_recut_ad_list_empty_when_no_markers(monkeypatch):
    monkeypatch.setattr(processing.db, 'get_episode', lambda s, e: {'ad_markers_json': None})
    assert processing._build_recut_ad_list('slug', 'ep', [], 600.0, '', 0.80) == ([], [])
