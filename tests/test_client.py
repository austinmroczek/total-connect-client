"""Test TotalConnectClient."""

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import requests
import requests_mock
from common import create_http_client
from const import (
    HTTP_RESPONSE_CONFIG,
    HTTP_RESPONSE_TOKEN,
    LOCATION_ID,
    PANEL_STATUS_DISARMED,
    RESPONSE_UNKNOWN,
    REST_RESULT_LOGOUT,
    REST_RESULT_PARTITIONS_CONFIG,
    REST_RESULT_PARTITIONS_ZONES,
    REST_RESULT_SESSION_DETAILS,
    SECURITY_DEVICE_ID,
)
from oauthlib.oauth2 import OAuth2Error
from pytest import raises

from total_connect_client.client import TotalConnectClient
from total_connect_client.const import (
    AUTH_CONFIG_ENDPOINT,
    AUTH_TOKEN_ENDPOINT,
    HTTP_API_LOGOUT,
    HTTP_API_SESSION_DETAILS_ENDPOINT,
    _ResultCode,
    make_http_endpoint,
)
from total_connect_client.exceptions import (
    AuthenticationError,
    BadResultCodeError,
    FailedToBypassZone,
    FeatureNotSupportedError,
    InvalidSessionError,
    RetryableTotalConnectError,
    ServiceUnavailable,
    TotalConnectError,
    UsercodeInvalid,
    UsercodeUnavailable,
)


def _response(code: int) -> dict:
    """Build a minimal API response dict with the given ResultCode."""
    return {"ResultCode": code, "ResultData": "test"}


@pytest.mark.parametrize(
    "code, expected",
    [
        (_ResultCode.SUCCESS.value, nullcontext()),
        (_ResultCode.INVALID_SESSION.value, raises(InvalidSessionError)),
        (_ResultCode.INVALID_SESSIONID.value, raises(InvalidSessionError)),
        (_ResultCode.CONNECTION_ERROR.value, raises(RetryableTotalConnectError)),
        (_ResultCode.FAILED_TO_CONNECT.value, raises(RetryableTotalConnectError)),
        (_ResultCode.CANNOT_CONNECT.value, raises(RetryableTotalConnectError)),
        (_ResultCode.BAD_OBJECT_REFERENCE.value, raises(RetryableTotalConnectError)),
    ],
)
def test_raise_for_retry(code, expected):
    client = create_http_client()
    with expected:
        client._raise_for_retry(_response(code))


@pytest.mark.parametrize(
    "code, expected",
    [
        (_ResultCode.SUCCESS.value, nullcontext()),
        (_ResultCode.ARM_SUCCESS.value, nullcontext()),
        (_ResultCode.BAD_USER_OR_PASSWORD.value, raises(AuthenticationError)),
        (_ResultCode.AUTHENTICATION_FAILED.value, raises(AuthenticationError)),
        (_ResultCode.ACCOUNT_LOCKED.value, raises(AuthenticationError)),
        (_ResultCode.USER_CODE_UNAVAILABLE.value, raises(UsercodeUnavailable)),
        (_ResultCode.USER_CODE_INVALID.value, raises(UsercodeInvalid)),
        (_ResultCode.FEATURE_NOT_SUPPORTED.value, raises(FeatureNotSupportedError)),
        (_ResultCode.FAILED_TO_BYPASS_ZONE.value, raises(FailedToBypassZone)),
        (_ResultCode.COMMAND_FAILED.value, raises(BadResultCodeError)),
    ],
)
def test_raise_for_resultcode(code, expected):
    client = create_http_client()
    with expected:
        client.raise_for_resultcode(_response(code))


# --- _request_with_retries ---


def test_request_with_retries_retryable_error_then_success():
    """RetryableTotalConnectError is retried; covers first-request and subsequent log branches."""
    client = create_http_client()
    attempts = 0

    def do_request():
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise RetryableTotalConnectError("transient error")
        return _response(_ResultCode.SUCCESS.value)

    with patch("total_connect_client.client.time.sleep") as mock_sleep:
        result = client._request_with_retries(do_request, "test request")

    assert result == _response(_ResultCode.SUCCESS.value)
    assert attempts == 3
    assert mock_sleep.call_count == 2


def test_request_with_retries_retryable_error_exhausted():
    """RetryableTotalConnectError is re-raised once all retries are consumed."""
    client = create_http_client()

    def do_request():
        raise RetryableTotalConnectError("permanent error")

    with patch("total_connect_client.client.time.sleep"), raises(RetryableTotalConnectError):
        client._request_with_retries(do_request, "test request")


def test_request_with_retries_invalid_session_triggers_reauth():
    """InvalidSessionError causes re-authentication before the next attempt."""
    client = create_http_client()
    attempts = 0

    def do_request():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise InvalidSessionError("session expired")
        return _response(_ResultCode.SUCCESS.value)

    with patch.object(client, "authenticate") as mock_auth:
        result = client._request_with_retries(do_request, "test request")

    mock_auth.assert_called_once()
    assert result == _response(_ResultCode.SUCCESS.value)


def test_request_with_retries_invalid_session_exhausted():
    """ServiceUnavailable is raised when re-authentication keeps failing."""
    client = create_http_client()

    def do_request():
        raise InvalidSessionError("session never valid")

    with patch.object(client, "authenticate"), raises(ServiceUnavailable):
        client._request_with_retries(do_request, "test request")


# --- http_request ---


def test_http_request_no_oauth_session():
    """TotalConnectError is raised immediately when the OAuth session is None."""
    client = create_http_client()
    client._oauth_session = None
    with raises(TotalConnectError):
        client.http_request(endpoint="https://example.com", method="GET")


def test_http_request_401_triggers_reauth():
    """HTTP 401 raises InvalidSessionError, triggering re-authentication via _request_with_retries."""
    client = create_http_client()
    mock_response = MagicMock(ok=False, status_code=401)
    with (
        patch.object(client._oauth_session, "request", return_value=mock_response),
        patch.object(client, "authenticate"),
        raises(ServiceUnavailable),
    ):
        client.http_request(endpoint="https://example.com", method="GET")


def test_http_request_5xx_triggers_retry():
    """HTTP 5xx raises RetryableTotalConnectError, which is retried then re-raised."""
    client = create_http_client()
    mock_response = MagicMock(ok=False, status_code=500)
    with (
        patch.object(client._oauth_session, "request", return_value=mock_response),
        patch("total_connect_client.client.time.sleep"),
        raises(RetryableTotalConnectError),
    ):
        client.http_request(endpoint="https://example.com", method="GET")


def tests_logout():
    """Test log_out."""
    client = create_http_client()
    assert client.is_logged_in() is True

    with requests_mock.Mocker() as rm:
        # first give an error
        rm.post(HTTP_API_LOGOUT, json=RESPONSE_UNKNOWN)
        with raises(TotalConnectError):
            client.log_out()
        assert client.is_logged_in() is True

        # then give success
        rm.post(HTTP_API_LOGOUT, json=REST_RESULT_LOGOUT)
        client.log_out()
        assert client.is_logged_in() is False


def test_get_configuration_connection_reset_all_fail():
    """Test that ServiceUnavailable is raised when ConnectionResetError persists across all retries.

    Regression test for https://github.com/home-assistant/core/issues/174207
    """
    with requests_mock.Mocker() as rm:
        rm.get(
            AUTH_CONFIG_ENDPOINT,
            exc=requests.exceptions.ConnectionError("Connection reset by peer"),
        )
        with raises(ServiceUnavailable):
            TotalConnectClient("username", "password", {}, retry_delay=0)


def test_get_configuration_connection_reset_then_success():
    """Test that a transient ConnectionResetError on _get_configuration is retried and recovers.

    Regression test for https://github.com/home-assistant/core/issues/174207
    """
    with requests_mock.Mocker() as rm:
        rm.get(
            AUTH_CONFIG_ENDPOINT,
            response_list=[
                {"exc": requests.exceptions.ConnectionError("Connection reset by peer")},
                {"json": HTTP_RESPONSE_CONFIG},
            ],
        )
        rm.post(AUTH_TOKEN_ENDPOINT, json=HTTP_RESPONSE_TOKEN)
        rm.get(HTTP_API_SESSION_DETAILS_ENDPOINT, json=REST_RESULT_SESSION_DETAILS)
        rm.get(
            make_http_endpoint(
                f"api/v1/locations/{LOCATION_ID}/devices/{SECURITY_DEVICE_ID}/partitions/config"
            ),
            json=REST_RESULT_PARTITIONS_CONFIG,
        )
        rm.get(
            make_http_endpoint(f"api/v1/locations/{LOCATION_ID}/partitions/zones/0"),
            json=REST_RESULT_PARTITIONS_ZONES,
        )
        rm.get(
            make_http_endpoint(f"api/v3/locations/{LOCATION_ID}/partitions/fullStatus"),
            json=PANEL_STATUS_DISARMED,
        )

        client = TotalConnectClient("username", "password", {LOCATION_ID: "1234"}, retry_delay=0)
        assert client.is_logged_in() is True


# --- _get_configuration error paths ---


def test_get_configuration_non_ok_response():
    """Non-OK HTTP response from the config endpoint raises ServiceUnavailable."""
    client = create_http_client()
    mock_response = MagicMock(ok=False, status_code=503)
    with (
        patch.object(client._raw_http_session, "get", return_value=mock_response),
        raises(ServiceUnavailable),
    ):
        client._get_configuration()


def test_get_configuration_invalid_json():
    """ValueError from response.json() raises ServiceUnavailable."""
    client = create_http_client()
    mock_response = MagicMock(ok=True)
    mock_response.json.side_effect = ValueError("no json")
    with (
        patch.object(client._raw_http_session, "get", return_value=mock_response),
        raises(ServiceUnavailable),
    ):
        client._get_configuration()


def test_get_configuration_malformed_config():
    """Config JSON that is missing required keys raises ServiceUnavailable."""
    client = create_http_client()
    mock_response = MagicMock(ok=True)
    mock_response.json.return_value = {"AppConfig": []}  # IndexError on [0]
    with (
        patch.object(client._raw_http_session, "get", return_value=mock_response),
        raises(ServiceUnavailable),
    ):
        client._get_configuration()


# --- _request_token / authenticate ---


def test_request_token_bad_credentials():
    """OAuth2Error with a BAD_USER_OR_PASSWORD payload raises AuthenticationError
    and permanently marks credentials as invalid."""
    client = create_http_client()
    bad_result = {"ResultCode": _ResultCode.BAD_USER_OR_PASSWORD.value, "ResultData": "test"}
    with (
        patch("total_connect_client.client.OAuth2Session.fetch_token", side_effect=OAuth2Error()),
        patch("total_connect_client.client.json.loads", return_value=bad_result),
        raises(AuthenticationError),
    ):
        client._request_token()

    assert client._invalid_credentials is True
    assert client._logged_in is False


def test_authenticate_raises_immediately_when_invalid_credentials():
    """authenticate() short-circuits with AuthenticationError when credentials already failed."""
    client = create_http_client()
    client._invalid_credentials = True
    with raises(AuthenticationError):
        client.authenticate()


# --- load_details ---


def test_load_details_already_loaded_skips_fetch():
    """Already-loaded location is skipped — covers the False branch of 'if not loaded'."""
    client = create_http_client()
    assert all(client._location_details.values())
    with patch.object(client._locations[LOCATION_ID], "get_partition_details") as mock_gpd:
        client.load_details()
    mock_gpd.assert_not_called()


def test_load_details_exception_triggers_retry():
    """Exception during detail fetch is caught and the location loads successfully on retry."""
    client = create_http_client()
    client._location_details[LOCATION_ID] = False
    location = client._locations[LOCATION_ID]
    call_count = 0

    def fail_once() -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient")

    with (
        patch.object(location, "get_partition_details", side_effect=fail_once),
        patch.object(location, "get_zone_details"),
        patch.object(location, "get_panel_meta_data"),
    ):
        client.load_details()

    assert call_count == 2
    assert client._location_details[LOCATION_ID] is True


def test_load_details_retries_exhausted_logs_warning():
    """When retries are exhausted, a warning is logged and no exception propagates."""
    client = create_http_client()
    client._location_details[LOCATION_ID] = False
    location = client._locations[LOCATION_ID]

    with (
        patch.object(location, "get_partition_details", side_effect=RuntimeError("always fails")),
        patch("total_connect_client.client.LOGGER") as mock_logger,
    ):
        client.load_details(retries=0)

    mock_logger.warning.assert_called_once()
    assert client._location_details[LOCATION_ID] is False


# --- _get_session_details ---


def test_get_session_details_no_locations_raises():
    """_get_session_details raises TotalConnectError when the response contains no locations."""
    client = create_http_client()
    client._locations.clear()
    session_result = {
        "ModuleFlags": "Security=1",
        "UserInfo": {
            "UserID": 123,
            "Username": "test",
            "UserFeatureList": "Master=0,User Administration=0,Configuration Administration=0",
        },
        "Locations": [],
    }
    with (
        patch.object(client, "http_request", return_value={"SessionDetailsResult": session_result}),
        raises(TotalConnectError),
    ):
        client._get_session_details()


# --- __init__, utility methods, and small branches ---


def test_init_load_details_false():
    """load_details=False skips loading partition/zone details during __init__."""
    with requests_mock.Mocker() as rm:
        rm.get(AUTH_CONFIG_ENDPOINT, json=HTTP_RESPONSE_CONFIG)
        rm.post(AUTH_TOKEN_ENDPOINT, json=HTTP_RESPONSE_TOKEN)
        rm.get(HTTP_API_SESSION_DETAILS_ENDPOINT, json=REST_RESULT_SESSION_DETAILS)
        client = TotalConnectClient(
            "username", "password", {LOCATION_ID: "1234"}, load_details=False
        )
    assert client._location_details[LOCATION_ID] is False


def test_times_as_string():
    """times_as_string returns a formatted string containing timing info."""
    client = create_http_client()
    result = client.times_as_string()
    assert "total running time" in result
    assert "total-connect-client time info" in result


def test_http_request_other_status_returns_json():
    """HTTP status not 401 or retry codes falls through and returns the JSON body."""
    client = create_http_client()
    mock_response = MagicMock(ok=False, status_code=403)
    mock_response.json.return_value = {"ResultCode": 0, "ResultData": "test"}
    with patch.object(client._oauth_session, "request", return_value=mock_response):
        result = client.http_request(endpoint="https://example.com", method="GET")
    assert result == {"ResultCode": 0, "ResultData": "test"}


def test_log_out_already_logged_out():
    """log_out is a no-op when the client is not logged in."""
    client = create_http_client()
    client._logged_in = False
    with patch.object(client._oauth_session, "request") as mock_request:
        client.log_out()
    mock_request.assert_not_called()


def test_log_out_arm_success_result_code_raises():
    """ResultCode ARM_SUCCESS passes raise_for_resultcode but is non-zero → TotalConnectError."""
    client = create_http_client()
    arm_success = {"ResultCode": _ResultCode.ARM_SUCCESS.value, "ResultData": "test"}
    with (
        patch.object(client, "http_request", return_value=arm_success),
        raises(TotalConnectError),
    ):
        client.log_out()


def test_get_number_locations():
    """get_number_locations returns the count of configured locations."""
    client = create_http_client()
    assert client.get_number_locations() == 1


def test_make_locations_no_usercode_uses_default():
    """Location without a matching usercode is assigned DEFAULT_USERCODE ('-1')."""
    client = create_http_client()
    client._locations.clear()
    client._location_details.clear()
    client.usercodes = {}
    location_info = REST_RESULT_SESSION_DETAILS["SessionDetailsResult"]["Locations"][0]
    client._make_locations({"Locations": [location_info]})
    assert LOCATION_ID in client._locations
    assert client._locations[LOCATION_ID].usercode == "-1"
