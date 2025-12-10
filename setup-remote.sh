#!/bin/bash

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <machine_name>"
    echo "Example: $0 big3"
    exit 1
fi

MACHINE="$1"
REMOTE="ubuntu@${MACHINE}"

echo "==> Installing unison on ${MACHINE}..."
ssh "${REMOTE}" "sudo apt install -y unison"

echo "==> Running unison sync for ${MACHINE}..."
while true; do
    if unison "${MACHINE}" 2>&1 | tee /dev/tty | grep -q "Synchronization complete"; then
        echo "==> Unison sync completed successfully"
        break
    else
        echo "==> Unison sync failed, retrying in 2 seconds..."
        sleep 2
    fi
done

echo "==> Installing uv on ${MACHINE}..."
ssh "${REMOTE}" "sudo snap install astral-uv --classic"

echo "==> Appending env.md to remote .bashrc..."
cat env.md | ssh "${REMOTE}" "cat >> ~/.bashrc"

echo "==> Done setting up ${MACHINE}"