#!/usr/bin/env bash
# Compatibility launcher. The full Pondok Huda-style manager lives in dev.sh.
set -e
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (($#)); then
    exec "$ROOT/dev.sh" "$@"
else
    exec "$ROOT/dev.sh" tui
fi
