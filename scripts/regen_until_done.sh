#!/usr/bin/env bash
# Retry a skill generation until it publishes.
#
#   scripts/regen_until_done.sh <spec.json> <name> [max_attempts=10] [pause_seconds=300]
#
# A generation is all-or-nothing per attempt, and the LLM gateway freezes for 10-20 minutes at a
# time (2026-10-20: one request took 992 s, the same request a minute later 8 s). Each attempt
# runs setup_cli --regenerate, which resumes from the stage cache a failed attempt left behind
# (skills/generation/stage_cache.py), so a retry pays only for the stages still missing.
#
# HARNESS_SKILLS_FOLDER is honored (generate into a scratch folder); HARNESS_CONFIG selects a
# harness.yaml other than config/harness.yaml. One line per attempt is appended to
# logs/regen-<name>.log: timestamp, attempt, exit code, last error line.
set -uo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <spec.json> <name> [max_attempts=10] [pause_seconds=300]" >&2
    exit 2
fi
SPEC="$(realpath "$1")"  # absolute: the script cds to the repo root below
NAME="$2"
MAX="${3:-10}"
PAUSE="${4:-300}"
CONFIG="${HARNESS_CONFIG:-config/harness.yaml}"

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
mkdir -p logs
LOG="logs/regen-$NAME.log"

for attempt in $(seq 1 "$MAX"); do
    out="$(mktemp)"
    uv run python -m simulation_harness.setup_cli "$SPEC" --name "$NAME" --regenerate \
        --config "$CONFIG" >"$out" 2>&1
    rc=$?
    last="$(grep -E 'ERROR|caused by' "$out" | tail -n 1)"
    printf '%s attempt=%s/%s rc=%s %s\n' "$(date '+%F %T')" "$attempt" "$MAX" "$rc" "$last" >>"$LOG"
    if [ "$rc" -eq 0 ]; then
        echo "published: $(tail -n 1 "$out")"
        rm -f "$out"
        exit 0
    fi
    echo "attempt $attempt/$MAX failed (rc=$rc): $last" >&2
    rm -f "$out"
    [ "$attempt" -lt "$MAX" ] && sleep "$PAUSE"
done
echo "gave up after $MAX attempts — see $LOG" >&2
exit 1
