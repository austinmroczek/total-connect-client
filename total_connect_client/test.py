"""Test your system from the command line."""

import sys
import logging
from pprint import pprint
import TotalConnectClient

logging.basicConfig(filename="test.log", level=logging.DEBUG)

if len(sys.argv) != 4:
    print("usage:  python3 test.py username password usercode \n")
    pring("usercode = -1 is a good default")
    sys.exit()


print(f"using username: {sys.argv[1]}")
print(f"using password: {sys.argv[2]}")
print(f"using usercode: {sys.argv[3]}")


print("\n\n\n")

tc = TotalConnectClient.TotalConnectClient(
    username=sys.argv[1], password=sys.argv[2], usercode=sys.argv[3], auto_bypass_battery=True
)
location_id = next(iter(tc.locations))

print("\n\n\n")

print("--- panel meta data ---")
meta_data = tc.get_panel_meta_data(location_id)
pprint(meta_data)

print("\n\n\n")

print(tc.locations[location_id])

for z in tc.locations[location_id].zones:
    print(tc.locations[location_id].zones[z])

print("\nCustomArmSettings\n\n")
pprint(tc.get_custom_arm_settings(location_id))

print("\n\n\n")
tc.arm_custom(1,location_id)