#!/bin/bash
# Tiny ad-hoc remote-exec helper: ./sshx.sh --vm vm03 '<remote command>'
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$(dirname "$0")/lib/common.sh"
bootstrap "$@"
set -- ${REMAINING_ARGS[@]+"${REMAINING_ARGS[@]}"}
ssh_cmd "$@"
