#!/bin/bash
# =============================================================================
# profiles/c2d.sh — VM identity for the "c2d" target (shared RTX 5090 box).
# =============================================================================
# Sourced by lib/common.sh::bootstrap when --vm c2d is used. This is a SHARED,
# multi-user machine (c2d-s12): 2x RTX 5090 32GB, co-located with colleagues'
# jobs. We live under /projetos/efficient-llms/andrew (world-writable shared
# project area, setgid group efficient-llms) — NOT a per-user ssd mount, so we
# override REMOTE_SSD_BASE. Uses a dedicated key (vm_c2d_s12), different port.
#
# GPU etiquette: pin CUDA_VISIBLE_DEVICES=1 and bound vllm memory so we never
# evict the co-located training job on GPU0/GPU1. See run_eval c2d overrides.
# =============================================================================

VM_USER="${VM_USER:-andrew.carvalho}"
VM_HOST="${VM_HOST:-200.144.192.77}"
VM_PORT="${VM_PORT:-1232}"

# No sudo on this box and /projetos is root-owned, so we live in our own home
# (same 7.2T root partition, private, not inside anyone else's work).
REMOTE_SSD_BASE="${REMOTE_SSD_BASE:-/home/andrew.carvalho/dynamic_graphqa}"

# Dedicated key for this box (the shared vm_key does not work here).
SSH_KEY="${SSH_KEY:-$HOME/.ssh/vm_c2d_s12}"
