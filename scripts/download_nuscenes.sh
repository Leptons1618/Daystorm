#!/usr/bin/env bash
# nuScenes mini (~4 GB) + the CAN bus expansion (~500 MB).
# Both need a free nuScenes account; export the cookie or download by hand
# from https://www.nuscenes.org/download and drop the tarballs in data/raw/.
set -euo pipefail
DEST="${1:-data/raw}"
mkdir -p "${DEST}"

echo "Expected in ${DEST}:"
echo "  v1.0-mini.tgz          ~4.0 GB   camera, radar, lidar, ego pose, annotations"
echo "  can_bus.zip            ~0.5 GB   steering, throttle, brake, yaw, wheel speeds"
echo
for f in v1.0-mini.tgz can_bus.zip; do
  [ -f "${DEST}/${f}" ] || { echo "MISSING: ${DEST}/${f}"; MISSING=1; }
done
[ "${MISSING:-0}" = "1" ] && { echo "download the files above, then re-run"; exit 1; }

tar -xzf "${DEST}/v1.0-mini.tgz" -C "${DEST}"
unzip -qo "${DEST}/can_bus.zip" -d "${DEST}"
echo "--> extracted to ${DEST}; point DAYSTORM_NUSCENES at it"
