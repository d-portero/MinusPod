"""Integration tests for per-template scoreThreshold PATCH endpoint (Task A1)."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

os.environ.setdefault('MINUSPOD_DATA_DIR', tempfile.mkdtemp(prefix='cue-thr-test-'))


def _csrf(app_client):
    with app_client.session_transaction() as sess:
        sess['authenticated'] = True
    app_client.get('/api/v1/auth/status')
    cookie = app_client.get_cookie('minuspod_csrf')
    return {'X-CSRF-Token': cookie.value} if cookie else {}


def _seed_template(db, slug_suffix, rng_seed):
    from audio_analysis.cue_features import N_COEFFS, serialize_mfcc, pcm_to_int16_bytes
    rng = np.random.default_rng(rng_seed)
    mfcc = rng.standard_normal((10, N_COEFFS)).astype(np.float32)
    pcm = rng.standard_normal(1600).astype(np.float32)
    pid = db.create_podcast(
        f'thr-{slug_suffix}', f'http://x/{slug_suffix}.xml', f'Feed {slug_suffix}')
    tid = db.create_cue_template(
        podcast_id=pid, cue_type='ad_break_boundary',
        source_episode_id='ep1', source_offset_s=1.0, duration_s=0.5,
        sample_rate=16000, n_coeffs=N_COEFFS,
        mfcc_blob=serialize_mfcc(mfcc),
        pcm_blob=pcm_to_int16_bytes(np.clip(pcm, -1, 1)),
        pcm_sample_rate=16000,
    )
    return tid


def test_patch_score_threshold_valid(app_client):
    """PATCH scoreThreshold=0.75 is accepted and echoed in the response."""
    from api import get_database
    db = get_database()
    headers = _csrf(app_client)
    tid = _seed_template(db, 'a', 20)

    resp = app_client.patch(
        f'/api/v1/cue-templates/{tid}',
        json={'scoreThreshold': 0.75},
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert abs(data['template']['scoreThreshold'] - 0.75) < 1e-6


def test_patch_score_threshold_null_clears(app_client):
    """PATCH scoreThreshold=null clears the column; response has null."""
    from api import get_database
    db = get_database()
    headers = _csrf(app_client)
    tid = _seed_template(db, 'b', 21)
    db.update_cue_template(tid, score_threshold=0.8)

    resp = app_client.patch(
        f'/api/v1/cue-templates/{tid}',
        json={'scoreThreshold': None},
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['template']['scoreThreshold'] is None


def test_patch_score_threshold_above_max_rejected(app_client):
    """PATCH scoreThreshold=1.0 (> 0.99) returns 400."""
    from api import get_database
    db = get_database()
    headers = _csrf(app_client)
    tid = _seed_template(db, 'c', 22)

    resp = app_client.patch(
        f'/api/v1/cue-templates/{tid}',
        json={'scoreThreshold': 1.0},
        headers=headers,
    )
    assert resp.status_code == 400


def test_patch_score_threshold_below_min_rejected(app_client):
    """PATCH scoreThreshold=-0.1 (< 0) returns 400."""
    from api import get_database
    db = get_database()
    headers = _csrf(app_client)
    tid = _seed_template(db, 'd', 23)

    resp = app_client.patch(
        f'/api/v1/cue-templates/{tid}',
        json={'scoreThreshold': -0.1},
        headers=headers,
    )
    assert resp.status_code == 400


def test_patch_score_threshold_non_numeric_rejected(app_client):
    """PATCH scoreThreshold='bad' returns 400."""
    from api import get_database
    db = get_database()
    headers = _csrf(app_client)
    tid = _seed_template(db, 'e', 24)

    resp = app_client.patch(
        f'/api/v1/cue-templates/{tid}',
        json={'scoreThreshold': 'bad'},
        headers=headers,
    )
    assert resp.status_code == 400


def test_patch_score_threshold_below_noise_floor_rejected(app_client):
    """PATCH scoreThreshold=0.10 (< 0.30 noise floor) returns 400."""
    from api import get_database
    db = get_database()
    headers = _csrf(app_client)
    tid = _seed_template(db, 'f', 25)

    resp = app_client.patch(
        f'/api/v1/cue-templates/{tid}',
        json={'scoreThreshold': 0.10},
        headers=headers,
    )
    assert resp.status_code == 400


def test_patch_score_threshold_at_noise_floor_accepted(app_client):
    """PATCH scoreThreshold=0.30 (exactly the floor) is accepted."""
    from api import get_database
    db = get_database()
    headers = _csrf(app_client)
    tid = _seed_template(db, 'g', 26)

    resp = app_client.patch(
        f'/api/v1/cue-templates/{tid}',
        json={'scoreThreshold': 0.30},
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert abs(data['template']['scoreThreshold'] - 0.30) < 1e-6


def test_patch_score_threshold_bool_rejected(app_client):
    """PATCH scoreThreshold=True (bool) returns 400; would silently coerce to 1.0."""
    from api import get_database
    db = get_database()
    headers = _csrf(app_client)
    tid = _seed_template(db, 'h', 27)

    resp = app_client.patch(
        f'/api/v1/cue-templates/{tid}',
        json={'scoreThreshold': True},
        headers=headers,
    )
    assert resp.status_code == 400
