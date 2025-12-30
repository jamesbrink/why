#!/usr/bin/env bash
# Common typo leading to "command not found"

echo "Checking system status..."

# Typo: should be "systemctl" not "systemclt"
systemclt status nginx

echo "Done!"
