#!/bin/bash
# =============================================================================
# profiles/vm02.sh — VM identity for the "vm02" target.
# =============================================================================
# Sourced by lib/common.sh::bootstrap when --vm vm02 is used. Same lab, same
# SSH key and port as vm03; only the host/user differ. config.sh derives all
# remote paths from this identity.
# =============================================================================

VM_USER="${VM_USER:-vm02}"
VM_HOST="${VM_HOST:-143.107.165.249}"
VM_PORT="${VM_PORT:-5022}"

# SSD base mirrors the username (=> /media/vm02/ssd1T/andrew). If vm02's SSD
# mounts elsewhere, uncomment and set the correct base — every remote path in
# config.sh then follows automatically.
# REMOTE_SSD_BASE="${REMOTE_SSD_BASE:-/media/vm02/ssd1T/andrew}"
