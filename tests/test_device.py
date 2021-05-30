"""Test TotalConnectDevice."""

from device import TotalConnectDevice

from const import DEVICE_INFO_BASIC_1

def tests_init():
    """Test __init__()."""
    test_device = TotalConnectDevice(DEVICE_INFO_BASIC_1)
    assert test_device.id == DEVICE_INFO_BASIC_1["DeviceID"]
