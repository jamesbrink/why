#!/usr/bin/env bash
# A script that triggers unbound variable error with set -u

set -euo pipefail

echo "Starting configuration..."

# Oops, DATABASE_URL was never set
echo "Connecting to: $DATABASE_URL"
echo "Done!"
