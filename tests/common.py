"""Common test code."""
from unittest.mock import patch

import requests_mock
from const import (HTTP_RESPONSE_CONFIG, HTTP_RESPONSE_TOKEN, LOCATION_ID,
                   PANEL_STATUS_DISARMED, REST_RESULT_PARTITIONS_CONFIG,
                   REST_RESULT_PARTITIONS_ZONES, REST_RESULT_SESSION_DETAILS,
                   SECURITY_DEVICE_ID)

from total_connect_client.client import TotalConnectClient
from total_connect_client.const import (AUTH_CONFIG_ENDPOINT,
                                        AUTH_TOKEN_ENDPOINT,
                                        HTTP_API_SESSION_DETAILS_ENDPOINT,
                                        make_http_endpoint)


def create_http_client(status=PANEL_STATUS_DISARMED):
    """Return a TotalConnectClient that appears to be logged in with given state.
    
    Status is a fullStatus response.  Default is disarmed.
    """

    with requests_mock.Mocker() as rm:
        rm.get(AUTH_CONFIG_ENDPOINT, json=HTTP_RESPONSE_CONFIG)
        rm.post(AUTH_TOKEN_ENDPOINT, json=HTTP_RESPONSE_TOKEN)
        rm.get(
            HTTP_API_SESSION_DETAILS_ENDPOINT,
            json=REST_RESULT_SESSION_DETAILS,
        )
        rm.get(make_http_endpoint(f"api/v1/locations/{LOCATION_ID}/devices/{SECURITY_DEVICE_ID}/partitions/config"), json=REST_RESULT_PARTITIONS_CONFIG)
        rm.get(make_http_endpoint(f"api/v1/locations/{LOCATION_ID}/partitions/zones/0"), json=REST_RESULT_PARTITIONS_ZONES)
        rm.get(make_http_endpoint(f"api/v3/locations/{LOCATION_ID}/partitions/fullStatus"), json=status)

        mock_client = TotalConnectClient("username", "password", {LOCATION_ID: "1234"})
        for location in mock_client.locations.values():
            location.get_panel_meta_data()

    return mock_client
