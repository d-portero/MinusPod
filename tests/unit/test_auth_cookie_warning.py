"""Tests for the plain-HTTP login warning that explains the silent
login-bounce loop a Secure session cookie causes over HTTP (issue #423)."""
from api.auth import _session_cookie_will_be_dropped


def test_warns_on_plain_http_with_secure_cookie():
    # The reported case: Secure cookie + plain-HTTP request, no https BASE_URL.
    assert _session_cookie_will_be_dropped(True, False, '') is True
    assert _session_cookie_will_be_dropped(True, False, 'http://host:8000') is True


def test_no_warn_when_cookie_not_secure():
    # Operator already set SESSION_COOKIE_SECURE=false; the cookie persists.
    assert _session_cookie_will_be_dropped(False, False, '') is False


def test_no_warn_on_secure_request():
    # Direct HTTPS or a trusted proxy with x_proto set: cookie persists.
    assert _session_cookie_will_be_dropped(True, True, '') is False


def test_no_warn_when_base_url_https():
    # TLS terminated upstream; request only looks insecure because the proxy
    # hop count is unset. Warning would be wrong advice here.
    assert _session_cookie_will_be_dropped(True, False, 'https://host') is False
    assert _session_cookie_will_be_dropped(True, False, 'HTTPS://host') is False
