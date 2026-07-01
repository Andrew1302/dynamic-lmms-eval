#!/bin/bash
# =============================================================================
# profiles/vm03.sh — VM identity for the "vm03" target (the default).
# =============================================================================
# Sourced by lib/common.sh::bootstrap when --vm vm03 (or no flag) is used.
# Only carries what is VM-specific; config.sh derives all remote paths from it.
# The ${VAR:-...} guards let an explicit shell env override still win.
# =============================================================================

VM_USER="${VM_USER:-vm03}"
VM_HOST="${VM_HOST:-143.107.165.250}"
VM_PORT="${VM_PORT:-5022}"

# SSD base mirrors the username (=> /media/vm03/ssd1T/andrew). Uncomment to
# override only if this VM's SSD mounts somewhere else.
# REMOTE_SSD_BASE="${REMOTE_SSD_BASE:-/media/vm03/ssd1T/andrew}"
