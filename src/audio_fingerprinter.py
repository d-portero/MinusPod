"""
Audio Fingerprinter - Chromaprint-based audio fingerprinting for ad detection.

Uses the Chromaprint library (via fpcalc binary) to generate audio fingerprints
that can identify identical or near-identical audio segments across episodes.
This is particularly effective for DAI (Dynamic Ad Insertion) ads that are
inserted as identical audio files.
"""
import ctypes
import logging
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import json

import numpy as np

try:
    import acoustid
except ImportError:
    acoustid = None

from utils.audio import get_audio_duration
from utils.subprocess_registry import tracked_run
from config import (
    FFMPEG_SHORT_TIMEOUT,
    FPCALC_TIMEOUT,
    FPCALC_TIMEOUT_FULL,
    SUBPROCESS_VERSION_PROBE,
    AUDIO_CUE_FP_WINDOW_SECONDS,
    AUDIO_CUE_FP_KEY_BITS,
    AUDIO_CUE_FP_KEY_SAMPLES,
    AUDIO_CUE_FP_MIN_GAP_SECONDS,
    AUDIO_CUE_FP_MAX_COUNT,
    AUDIO_CUE_FP_MAX_LEN_SECONDS,
    AUDIO_CUE_FP_MAX_ANCHORS,
    AUDIO_CUE_FP_MAX_CANDIDATES,
)

logger = logging.getLogger('podcast.fingerprint')

# Fingerprint matching threshold (0-1, lower = more strict)
# 0.65 allows for minor encoding differences while avoiding false positives
MATCH_THRESHOLD = 0.65

# Minimum duration for fingerprinting (seconds)
MIN_SEGMENT_DURATION = 5.0

# Fingerprint chunk size for sliding window search (seconds)
FINGERPRINT_CHUNK_SIZE = 10.0

# Step size for sliding window (seconds)
SLIDING_STEP_SIZE = 2.0

# Cap for the per-window slow scan when the full-file fast path fails.
# When fpcalc can't decode the audio end-to-end, the per-window scan uses
# the same fpcalc binary on each window and almost always produces zero
# new matches -- the only realistic save is a single bad frame midway. 90s
# is enough to catch that case without burning the 10-minute upper bound.
FALLBACK_SLOW_TIMEOUT = 90

# 256-entry table for vectorized population count over uint32 numpy arrays.
_POPCOUNT8 = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint16)


def _popcount32(x):
    """Vectorized population count for a uint32 numpy array."""
    return (_POPCOUNT8[x & 0xFF] + _POPCOUNT8[(x >> 8) & 0xFF]
            + _POPCOUNT8[(x >> 16) & 0xFF] + _POPCOUNT8[(x >> 24) & 0xFF])


# _window_similarity / _pair_similarity are the numpy-vectorized twins of the
# scalar AudioFingerprinter._calculate_similarity: the same bit-error-rate metric
# (1 - hamming(a XOR b) / bits) over uint32-masked Chromaprint ints. The np.uint32
# array dtype handles the signed-int masking that _calculate_similarity does with
# `& 0xFFFFFFFF`. Kept separate because these run the comparison over every offset
# at once, where the scalar version walks one pair in a Python loop.
def _window_similarity(arr, anchor, win):
    """Bit-similarity (0-1) of window [anchor, anchor+win) vs every start position."""
    m = len(arr) - win + 1
    diff = np.zeros(m, dtype=np.int64)
    for k in range(win):
        diff += _popcount32(arr[k:k + m] ^ arr[anchor + k])
    return 1.0 - diff / (win * 32)


def _pair_similarity(arr, a, b, length):
    """Bit-similarity (0-1) of the two length-``length`` windows at ``a`` and ``b``."""
    bad = int(_popcount32(arr[a:a + length] ^ arr[b:b + length]).sum())
    return 1.0 - bad / (length * 32)


def _greedy_hit_positions(sim, similarity, min_gap, claimed=None):
    """Walk a similarity curve left-to-right, taking one hit per ``min_gap`` run.

    Once a position clears ``similarity`` it is taken and the next ``min_gap``
    positions are skipped, so a single occurrence is counted once. ``claimed``,
    when given, suppresses positions an earlier candidate already owns.
    """
    hits = []
    p = 0
    m = len(sim)
    while p < m:
        if sim[p] >= similarity and (claimed is None or not claimed[p]):
            hits.append(p)
            p += min_gap
        else:
            p += 1
    return hits


def _count_window_matches(raw_ints, fp_duration, start_s, end_s, similarity):
    """Count how many times the [start_s, end_s] window recurs in the file.

    A self-match of a captured cue: 1 means it appears only where it was
    captured (a non-recurring, weak template); >=2 means it recurs and can
    bracket ad breaks. Pure function over the fpcalc ``-raw`` int array.
    """
    n = len(raw_ints)
    if n == 0 or fp_duration <= 0 or end_s <= start_s:
        return 0
    fps = n / fp_duration
    anchor = min(max(0, int(round(start_s * fps))), n - 1)
    win = min(max(4, int(round((end_s - start_s) * fps))), n - anchor)
    if win < 4:
        return 0
    arr = np.asarray(raw_ints, dtype=np.uint32)
    min_gap = max(1, int(round(AUDIO_CUE_FP_MIN_GAP_SECONDS * fps)))
    sim = _window_similarity(arr, anchor, win)
    return len(_greedy_hit_positions(sim, similarity, min_gap))


def _discover_repeats(raw_ints, fp_duration, similarity, min_count):
    """Find windows of a raw Chromaprint fingerprint that recur across the file.

    Pure function over the fpcalc ``-raw`` int array (no I/O). A short probe
    window seeds LSH buckets; each bucket's first member anchors a full
    self-Hamming scan, the matched segment is grown to its true length, and its
    whole extent is claimed so a long recurring block surfaces as one candidate
    rather than many fragments. Loudness-independent.

    Args:
        raw_ints: fpcalc ``-raw`` fingerprint as a list/array of ints (~8/sec).
        fp_duration: duration the fingerprint covers, in seconds.
        similarity: per-window bit-similarity (0-1) two occurrences must reach.
        min_count: minimum occurrences for a sound to be suggested.

    Returns:
        Candidate dicts {start, end, count} in descending recurrence order,
        capped at AUDIO_CUE_FP_MAX_CANDIDATES.
    """
    n = len(raw_ints)
    if n == 0 or fp_duration <= 0:
        return []
    fps = n / fp_duration
    win = max(4, int(round(AUDIO_CUE_FP_WINDOW_SECONDS * fps)))
    if n < win * 2:
        return []
    min_gap = max(1, int(round(AUDIO_CUE_FP_MIN_GAP_SECONDS * fps)))
    max_len = max(win, int(round(AUDIO_CUE_FP_MAX_LEN_SECONDS * fps)))
    step = max(1, win // 2)
    # Via int64 so signed/out-of-range ints wrap into uint32 (matching the
    # `& 0xFFFFFFFF` masking in _calculate_similarity) instead of warning.
    arr = np.asarray(raw_ints, dtype=np.int64).astype(np.uint32)

    # LSH seed: bucket each probe window by the top KEY_BITS of KEY_SAMPLES
    # evenly spaced subfingerprints, so windows of the same sound collide.
    samples = [int(j * (win - 1) / (AUDIO_CUE_FP_KEY_SAMPLES - 1))
               for j in range(AUDIO_CUE_FP_KEY_SAMPLES)]
    shift = 32 - AUDIO_CUE_FP_KEY_BITS
    buckets = {}
    for i in range(0, n - win + 1, step):
        key = tuple(int(arr[i + s]) >> shift for s in samples)
        buckets.setdefault(key, []).append(i)
    anchors = [members[0] for members in
               sorted(buckets.values(), key=lambda m: -len(m))
               if len(members) >= 2][:AUDIO_CUE_FP_MAX_ANCHORS]

    claimed = np.zeros(n, dtype=bool)
    candidates = []
    for anchor in anchors:
        if claimed[anchor]:
            continue
        hits = _greedy_hit_positions(
            _window_similarity(arr, anchor, win), similarity, min_gap, claimed)
        if not (min_count <= len(hits) <= AUDIO_CUE_FP_MAX_COUNT):
            continue
        # Reference the first matching occurrence (hits is ascending), not the
        # LSH bucket member, which can sit mid-run and even after hits[-1]; using
        # the smallest hit keeps every shifted index in [0, n) below.
        ref = hits[0]
        # The match usually lands mid-sound. Walk the whole occurrence set back
        # to the true onset so the candidate points at the sound's start (and its
        # claim absorbs earlier fragments of the same block).
        back = 0
        while (ref - (back + step) >= 0
               and all(_pair_similarity(arr, ref - back - step, h - back - step, win) >= similarity
                       for h in hits[1:])):
            back += step
        seg_hits = [h - back for h in hits]   # ascending; seg_hits[0] == ref - back
        seg_start = seg_hits[0]
        # Backward extension can walk into a region an earlier (stronger)
        # candidate already claimed; if so this is the same sound seen from a
        # weaker anchor -- drop it rather than emit an overlapping duplicate.
        if claimed[seg_start]:
            continue
        # Grow the segment forward while every occurrence keeps matching. The
        # largest occurrence (seg_hits[-1]) bounds the in-file check.
        length = win + back
        while length + step <= max_len and seg_hits[-1] + length + step <= n:
            if all(_pair_similarity(arr, seg_start, sh, length + step) >= similarity
                   for sh in seg_hits[1:]):
                length += step
            else:
                break
        for sh in seg_hits:
            claimed[max(0, sh - min_gap):min(sh + length + min_gap, n)] = True
        start_s = seg_start / fps
        candidates.append({
            'start': round(start_s, 2),
            'end': round(start_s + length / fps, 2),
            'count': len(hits),
        })
    candidates.sort(key=lambda c: -c['count'])
    return candidates[:AUDIO_CUE_FP_MAX_CANDIDATES]


@dataclass
class FingerprintMatch:
    """Represents a fingerprint match in an audio file."""
    pattern_id: int
    start: float
    end: float
    confidence: float
    sponsor: Optional[str] = None


@dataclass
class AudioFingerprint:
    """Represents an audio fingerprint."""
    fingerprint: str  # Raw chromaprint fingerprint
    duration: float
    pattern_id: Optional[int] = None


class AudioFingerprinter:
    """
    Audio fingerprinting using Chromaprint for identifying repeated ads.

    This class provides functionality to:
    - Generate fingerprints for audio segments
    - Compare fingerprints to find matches
    - Search for known ad fingerprints in new episodes
    """

    def __init__(self, db=None):
        """
        Initialize the audio fingerprinter.

        Args:
            db: Database instance for storing/retrieving fingerprints
        """
        self.db = db
        self._fpcalc_path = self._find_fpcalc()

    def _find_fpcalc(self) -> Optional[str]:
        """Find the fpcalc binary."""
        # Check common locations
        paths = [
            '/usr/bin/fpcalc',
            '/usr/local/bin/fpcalc',
            'fpcalc'  # In PATH
        ]

        for path in paths:
            try:
                result = subprocess.run(
                    [path, '-version'],
                    capture_output=True,
                    timeout=SUBPROCESS_VERSION_PROBE
                )
                if result.returncode == 0:
                    logger.debug(f"Found fpcalc at: {path}")
                    return path
            except (subprocess.SubprocessError, FileNotFoundError):
                continue

        logger.warning("fpcalc not found - audio fingerprinting disabled")
        return None

    def is_available(self) -> bool:
        """Check if audio fingerprinting is available."""
        return self._fpcalc_path is not None

    def generate_fingerprint(
        self,
        audio_path: str,
        start: float = 0,
        duration: float = None
    ) -> Optional[AudioFingerprint]:
        """
        Generate a fingerprint for an audio segment.

        Args:
            audio_path: Path to audio file
            start: Start time in seconds
            duration: Duration in seconds (None = entire file)

        Returns:
            AudioFingerprint or None if generation failed
        """
        if not self._fpcalc_path:
            return None

        try:
            # Build fpcalc command
            cmd = [self._fpcalc_path, '-json']

            # If we need a specific segment, extract it first
            if start > 0 or duration is not None:
                # Use ffmpeg to extract segment
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp_path = tmp.name

                try:
                    ffmpeg_cmd = [
                        'ffmpeg', '-y', '-i', audio_path,
                        '-ss', str(start),
                    ]
                    if duration:
                        ffmpeg_cmd.extend(['-t', str(duration)])
                    ffmpeg_cmd.extend([
                        '-ac', '1',  # Mono
                        '-ar', '16000',  # 16kHz
                        '-f', 'wav',
                        tmp_path
                    ])

                    tracked_run(
                        ffmpeg_cmd,
                        capture_output=True,
                        timeout=FFMPEG_SHORT_TIMEOUT,
                        check=True,
                    )

                    cmd.append(tmp_path)
                    result = tracked_run(
                        cmd,
                        capture_output=True,
                        timeout=FPCALC_TIMEOUT,
                    )
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            else:
                cmd.append(audio_path)
                result = tracked_run(
                    cmd,
                    capture_output=True,
                    timeout=FPCALC_TIMEOUT,
                )

            if result.returncode != 0:
                logger.warning(f"fpcalc failed: {result.stderr.decode()}")
                return None

            # Parse JSON output
            data = json.loads(result.stdout.decode())

            return AudioFingerprint(
                fingerprint=data.get('fingerprint', ''),
                duration=data.get('duration', duration or 0)
            )

        except subprocess.TimeoutExpired:
            logger.error("Fingerprint generation timed out")
            return None
        except Exception as e:
            logger.error(f"Fingerprint generation failed: {e}")
            return None

    def compare_fingerprints(
        self,
        fp1: str,
        fp2: str
    ) -> float:
        """
        Compare two fingerprints and return similarity score.

        Uses bit error rate comparison on the raw fingerprint data.

        Args:
            fp1: First fingerprint string
            fp2: Second fingerprint string

        Returns:
            Similarity score between 0 and 1
        """
        if acoustid is None:
            logger.warning("acoustid module not available for fingerprint comparison")
            return 0.0
        try:
            # decode_fingerprint expects bytes, not str (ctypes c_char pointer)
            if isinstance(fp1, str):
                fp1 = fp1.encode('utf-8')
            if isinstance(fp2, str):
                fp2 = fp2.encode('utf-8')

            # Decode fingerprints to integer arrays
            fp1_decoded = acoustid.chromaprint.decode_fingerprint(fp1)
            fp2_decoded = acoustid.chromaprint.decode_fingerprint(fp2)

            if not fp1_decoded or not fp2_decoded:
                return 0.0

            fp1_array = fp1_decoded[0]
            fp2_array = fp2_decoded[0]

            # Compare using bit error rate
            return self._calculate_similarity(fp1_array, fp2_array)

        except (TypeError, ctypes.ArgumentError) as e:
            logger.error(f"Fingerprint comparison failed (bad data): {e}")
            return -1.0
        except Exception as e:
            logger.error(f"Fingerprint comparison failed: {e}")
            return 0.0

    def _calculate_similarity(
        self,
        fp1: List[int],
        fp2: List[int],
        fp1_start: int = 0,
        fp1_end: int = 0
    ) -> float:
        """
        Calculate similarity between two fingerprint arrays using bit error rate.

        Scalar twin of the module-level _window_similarity / _pair_similarity
        (same metric); this one walks one slice pair in a Python loop for the
        ad-matching path, those vectorize it over every offset for cue discovery.

        Args:
            fp1: First fingerprint array
            fp2: Second fingerprint array
            fp1_start: Start index into fp1 (default 0)
            fp1_end: End index into fp1 (default 0 means len(fp1))

        Returns:
            Similarity score between 0 and 1
        """
        if not fp1 or not fp2:
            return 0.0

        if fp1_end == 0:
            fp1_end = len(fp1)

        # Use the shorter length for comparison
        min_len = min(fp1_end - fp1_start, len(fp2))
        if min_len <= 0:
            return 0.0

        # Count matching bits
        total_bits = 0
        matching_bits = 0

        for i in range(min_len):
            # Mask to 32 bits: fpcalc -raw emits signed ints, and
            # int.bit_count() counts bits of abs(value), not two's complement
            xor = (fp1[fp1_start + i] ^ fp2[i]) & 0xFFFFFFFF
            diff_bits = xor.bit_count()
            matching_bits += 32 - diff_bits
            total_bits += 32

        return matching_bits / total_bits if total_bits > 0 else 0.0

    def _generate_full_fingerprint(
        self, audio_path: str, timeout: int = FPCALC_TIMEOUT_FULL
    ) -> Optional[Tuple[List[int], float]]:
        """Generate raw fingerprint for entire audio file in one fpcalc call.

        Args:
            audio_path: Path to the audio file.
            timeout: fpcalc wall-clock cap. Defaults to the full-file budget; a
                caller on a request thread can pass a shorter bound so a stalled
                decode degrades gracefully instead of holding the request.

        Returns:
            Tuple of (raw_int_array, duration) or None on failure
        """
        if not self._fpcalc_path:
            return None

        try:
            cmd = [self._fpcalc_path, '-raw', '-json', '-length', '0', audio_path]
            result = tracked_run(cmd, capture_output=True, timeout=timeout)

            if result.returncode != 0:
                logger.warning(f"Full-file fpcalc failed: {result.stderr.decode()}")
                return None

            data = json.loads(result.stdout.decode())
            raw_ints = data.get('fingerprint', [])
            duration = data.get('duration', 0)

            if not raw_ints or not isinstance(raw_ints, list):
                return None

            logger.debug(f"Full-file fingerprint: {len(raw_ints)} ints for {duration:.1f}s")
            return (raw_ints, duration)

        except subprocess.TimeoutExpired:
            logger.error("Full-file fingerprint generation timed out")
            return None
        except Exception as e:
            logger.error(f"Full-file fingerprint generation failed: {e}")
            return None

    def discover_recurring_spots(self, audio_path, *, similarity, min_count):
        """Find recurring sounds in an episode as cue-template candidates.

        Generates one full-file raw Chromaprint fingerprint, then surfaces the
        windows that recur at least ``min_count`` times. Loudness-independent,
        so it catches level-matched stings the loudness scan misses.

        Returns candidate dicts {start, end, count} in descending recurrence
        order, or [] if fpcalc is unavailable or fails.
        """
        if not self._fpcalc_path:
            return []
        full_fp = self._generate_full_fingerprint(audio_path)
        if full_fp is None:
            logger.warning(
                "Cue candidate discovery: full-file fingerprint failed for %s",
                audio_path)
            return []
        raw_ints, fp_duration = full_fp
        candidates = _discover_repeats(raw_ints, fp_duration, similarity, min_count)
        logger.info(
            "Cue candidate discovery: %d candidates from %d subfingerprints (%.0fs)",
            len(candidates), len(raw_ints), fp_duration)
        return candidates

    def count_self_matches(self, audio_path, start_s, end_s, *, similarity):
        """Count how many times a captured cue window recurs in its episode.

        Used at template-create time to warn on a weak cue: 1 means the sound
        appears only where it was captured (it will not bracket ad breaks); >=2
        means it recurs. Returns 0 if fpcalc is unavailable or fails.

        Runs on the create request thread, so the fingerprint is bounded by the
        shorter FPCALC_TIMEOUT (not the full-file budget): a normal episode
        fingerprints in seconds, and a stalled decode gives up well before the
        proxy timeout, yielding 0 (no warning) rather than blocking the save.
        """
        if not self._fpcalc_path:
            return 0
        full_fp = self._generate_full_fingerprint(audio_path, timeout=FPCALC_TIMEOUT)
        if full_fp is None:
            return 0
        raw_ints, fp_duration = full_fp
        return _count_window_matches(raw_ints, fp_duration, start_s, end_s, similarity)

    def _decode_known_fingerprints(
        self,
        known_fingerprints: List[Tuple[int, str, float, str]]
    ) -> List[Tuple[int, List[int], float, str]]:
        """Decode known fingerprint strings to raw int arrays.

        Returns:
            List of (pattern_id, raw_int_array, duration, sponsor)
        """
        if acoustid is None:
            logger.warning("acoustid not available for fingerprint decoding")
            return []

        decoded = []
        for pattern_id, fp_str, duration, sponsor in known_fingerprints:
            try:
                fp_bytes = fp_str.encode('utf-8') if isinstance(fp_str, str) else fp_str
                result = acoustid.chromaprint.decode_fingerprint(fp_bytes)
                if result and result[0]:
                    decoded.append((pattern_id, result[0], duration, sponsor))
                else:
                    logger.warning(f"Could not decode fingerprint for pattern {pattern_id}")
            except Exception as e:
                logger.warning(f"Failed to decode fingerprint for pattern {pattern_id}: {e}")

        return decoded

    def _find_matches_fast(
        self,
        raw_ints: List[int],
        fp_duration: float,
        decoded_known: List[Tuple[int, List[int], float, str]],
        total_duration: float,
        timeout: int,
        cancel_event: Optional[threading.Event]
    ) -> List[FingerprintMatch]:
        """Fast fingerprint matching using pre-computed full-file fingerprint.

        Slides through the raw int array comparing slices against decoded
        known fingerprints. No subprocess calls -- pure Python array operations.
        """
        matches = []
        # fpcalc default sample rate produces ~8 ints/sec; compute actual from data
        ints_per_second = len(raw_ints) / fp_duration if fp_duration > 0 else 8.0
        scan_start_time = time.time()
        last_log_time = scan_start_time
        position = 0.0

        while position < total_duration - MIN_SEGMENT_DURATION:
            now = time.time()
            elapsed = now - scan_start_time

            if elapsed > timeout:
                logger.warning(
                    f"Fingerprint scan timed out after {elapsed:.0f}s "
                    f"at {position:.1f}s/{total_duration:.1f}s with {len(matches)} matches"
                )
                break

            if cancel_event and cancel_event.is_set():
                logger.info(f"Fingerprint scan cancelled at {position:.1f}s/{total_duration:.1f}s")
                break

            if now - last_log_time >= 60:
                pct = (position / total_duration) * 100
                logger.info(
                    f"Fingerprint scan progress: {position:.1f}s/{total_duration:.1f}s "
                    f"({pct:.0f}%), {len(matches)} matches, {elapsed:.0f}s elapsed"
                )
                last_log_time = now

            # Compute indices into raw_ints for current window (avoid list copy)
            start_idx = int(position * ints_per_second)
            end_idx = int((position + FINGERPRINT_CHUNK_SIZE) * ints_per_second)
            end_idx = min(end_idx, len(raw_ints))

            if end_idx - start_idx < 4:
                position += SLIDING_STEP_SIZE
                continue

            matched = False
            for pattern_id, known_ints, known_duration, sponsor in decoded_known:
                similarity = self._calculate_similarity(
                    raw_ints, known_ints, fp1_start=start_idx, fp1_end=end_idx
                )

                if similarity >= MATCH_THRESHOLD:
                    match = FingerprintMatch(
                        pattern_id=pattern_id,
                        start=position,
                        end=position + known_duration,
                        confidence=similarity,
                        sponsor=sponsor
                    )
                    matches.append(match)
                    logger.info(
                        f"Fingerprint match: pattern={pattern_id} "
                        f"at {position:.1f}s (confidence={similarity:.2f})"
                    )
                    position += known_duration
                    matched = True
                    break

            if not matched:
                position += SLIDING_STEP_SIZE

        matches = self._merge_overlapping_matches(matches)

        scan_elapsed = time.time() - scan_start_time
        logger.info(
            f"Fast fingerprint scan completed in {scan_elapsed:.1f}s, "
            f"found {len(matches)} matches"
        )

        return matches

    def find_matches(
        self,
        audio_path: str,
        known_fingerprints: List[Tuple[int, str, float, str]] = None,
        timeout: int = 600,
        cancel_event: Optional[threading.Event] = None
    ) -> List[FingerprintMatch]:
        """
        Search for known ad fingerprints in an audio file.

        Uses a sliding window approach to find matches at any position.

        Args:
            audio_path: Path to audio file to search
            known_fingerprints: List of (pattern_id, fingerprint, duration, sponsor)
                               If None, loads from database
            timeout: Maximum seconds to spend scanning (default 600s / 10 minutes).
                     Returns partial results if exceeded.
            cancel_event: Optional threading.Event; if set, scanning stops early.

        Returns:
            List of FingerprintMatch objects for found ads
        """
        if not self.is_available():
            return []

        # Load known fingerprints from database if not provided
        if known_fingerprints is None and self.db:
            known_fingerprints = self._load_fingerprints_from_db()

        if not known_fingerprints:
            return []

        matches = []
        broken_patterns = set()

        # Get total duration of audio
        total_duration = self._get_audio_duration(audio_path)
        if total_duration <= 0:
            return []

        logger.info(f"Searching {total_duration:.1f}s audio for {len(known_fingerprints)} known fingerprints")

        # Fast path: generate one full-file fingerprint and compare by slicing
        full_fp = self._generate_full_fingerprint(audio_path)
        if full_fp is not None:
            raw_ints, fp_duration = full_fp
            decoded_known = self._decode_known_fingerprints(known_fingerprints)
            if decoded_known:
                logger.info(
                    f"Using fast fingerprint scan "
                    f"({len(raw_ints)} ints, {len(decoded_known)} patterns)"
                )
                return self._find_matches_fast(
                    raw_ints, fp_duration, decoded_known, total_duration,
                    timeout, cancel_event
                )
            else:
                logger.warning("Could not decode known fingerprints, falling back to per-window scan")
        else:
            logger.warning("Full-file fingerprint failed, falling back to per-window scan")

        # Slow fallback: per-window subprocess scanning.
        # Cap separately at FALLBACK_SLOW_TIMEOUT (much shorter than the
        # full-file timeout). When the fast path fails because fpcalc can't
        # decode the audio source, the per-window scan uses the same fpcalc
        # and almost always produces zero new matches -- burning the full
        # 10-minute budget is wasted work. 90s is enough to catch the rare
        # case where the failure was a single bad frame.
        slow_timeout = min(timeout, FALLBACK_SLOW_TIMEOUT)
        scan_start_time = time.time()
        last_log_time = scan_start_time
        position = 0.0
        while position < total_duration - MIN_SEGMENT_DURATION:
            now = time.time()
            elapsed = now - scan_start_time

            # Timeout check
            if elapsed > slow_timeout:
                logger.warning(
                    f"Fingerprint scan timed out after {elapsed:.0f}s "
                    f"at {position:.1f}s/{total_duration:.1f}s with {len(matches)} matches"
                )
                break

            # Cancel check
            if cancel_event and cancel_event.is_set():
                logger.info(f"Fingerprint scan cancelled at {position:.1f}s/{total_duration:.1f}s")
                break

            # Progress logging every 60s
            if now - last_log_time >= 60:
                pct = (position / total_duration) * 100
                logger.info(
                    f"Fingerprint scan progress: {position:.1f}s/{total_duration:.1f}s "
                    f"({pct:.0f}%), {len(matches)} matches, {elapsed:.0f}s elapsed"
                )
                last_log_time = now
            # Bail out if all known fingerprints are broken/corrupt
            if len(broken_patterns) >= len(known_fingerprints):
                logger.info("All known fingerprints are broken/skipped, ending scan early")
                break

            # Generate fingerprint for current window
            chunk_fp = self.generate_fingerprint(
                audio_path,
                start=position,
                duration=FINGERPRINT_CHUNK_SIZE
            )

            if chunk_fp and chunk_fp.fingerprint:
                # Compare against known fingerprints
                for pattern_id, known_fp, known_duration, sponsor in known_fingerprints:
                    if pattern_id in broken_patterns:
                        continue

                    similarity = self.compare_fingerprints(
                        chunk_fp.fingerprint,
                        known_fp
                    )

                    if similarity < 0:
                        broken_patterns.add(pattern_id)
                        logger.warning(f"Skipping broken fingerprint pattern {pattern_id} for remaining audio")
                        if self.db:
                            try:
                                self.db.delete_audio_fingerprint(pattern_id)
                                logger.warning(f"Deleted corrupt fingerprint for pattern {pattern_id}")
                            except Exception as del_err:
                                logger.error(f"Failed to delete corrupt fingerprint {pattern_id}: {del_err}")
                        continue

                    if similarity >= MATCH_THRESHOLD:
                        # Found a match
                        match = FingerprintMatch(
                            pattern_id=pattern_id,
                            start=position,
                            end=position + known_duration,
                            confidence=similarity,
                            sponsor=sponsor
                        )
                        matches.append(match)
                        logger.info(
                            f"Fingerprint match: pattern={pattern_id} "
                            f"at {position:.1f}s (confidence={similarity:.2f})"
                        )
                        # Skip ahead past this match
                        position += known_duration
                        break
                else:
                    position += SLIDING_STEP_SIZE
            else:
                position += SLIDING_STEP_SIZE

        # Merge overlapping matches
        matches = self._merge_overlapping_matches(matches)

        return matches

    def _load_fingerprints_from_db(self) -> List[Tuple[int, str, float, str]]:
        """Load known fingerprints from database with sponsors (single JOIN query)."""
        if not self.db:
            return []

        try:
            fingerprints = self.db.get_all_fingerprints_with_sponsors()
            result = []
            for fp in fingerprints:
                # Fingerprint may be stored as bytes or string
                fp_data = fp.get('fingerprint', b'')
                if isinstance(fp_data, bytes):
                    fp_str = fp_data.decode('utf-8', errors='ignore')
                else:
                    fp_str = str(fp_data)

                result.append((
                    fp['pattern_id'],
                    fp_str,
                    fp['duration'],
                    fp.get('sponsor')
                ))
            return result
        except Exception as e:
            logger.error(f"Failed to load fingerprints from database: {e}")
            return []

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file in seconds.

        Delegates to utils.audio.get_audio_duration for consistent implementation.
        """
        duration = get_audio_duration(audio_path)
        return duration if duration is not None else 0.0

    def _merge_overlapping_matches(
        self,
        matches: List[FingerprintMatch]
    ) -> List[FingerprintMatch]:
        """Merge overlapping fingerprint matches."""
        if not matches:
            return []

        # Sort by start time
        matches.sort(key=lambda m: m.start)

        merged = []
        current = matches[0]

        for match in matches[1:]:
            # Check for overlap
            if match.start <= current.end + 1.0:  # 1s tolerance
                # Extend current match
                current = FingerprintMatch(
                    pattern_id=current.pattern_id,
                    start=current.start,
                    end=max(current.end, match.end),
                    confidence=max(current.confidence, match.confidence),
                    sponsor=current.sponsor or match.sponsor
                )
            else:
                merged.append(current)
                current = match

        merged.append(current)
        return merged

    def store_fingerprint(
        self,
        pattern_id: int,
        audio_path: str,
        start: float,
        end: float
    ) -> bool:
        """
        Generate and store a fingerprint for a detected ad segment.

        Args:
            pattern_id: ID of the ad pattern
            audio_path: Path to the episode audio
            start: Start time of the ad
            end: End time of the ad

        Returns:
            True if fingerprint was stored successfully
        """
        if not self.db or not self.is_available():
            return False

        duration = end - start
        if duration < MIN_SEGMENT_DURATION:
            logger.debug(f"Segment too short for fingerprinting: {duration:.1f}s")
            return False

        fingerprint = self.generate_fingerprint(audio_path, start, duration)
        if not fingerprint or not fingerprint.fingerprint:
            return False

        try:
            # Store fingerprint as bytes
            fp_bytes = fingerprint.fingerprint.encode('utf-8')
            self.db.create_audio_fingerprint(
                pattern_id=pattern_id,
                fingerprint=fp_bytes,
                duration=duration
            )
            logger.info(f"Stored fingerprint for pattern {pattern_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store fingerprint: {e}")
            return False
