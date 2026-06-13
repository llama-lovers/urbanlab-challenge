#!/bin/bash
# Usage: ./wait_for_db.sh <host> <port> [timeout_seconds]
HOST="${1:-postgres}"
PORT="${2:-5432}"
TIMEOUT="${3:-30}"
ELAPSED=0

echo "Waiting for $HOST:$PORT..."
until nc -z "$HOST" "$PORT" 2>/dev/null; do
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo "Timed out waiting for $HOST:$PORT"
        exit 1
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done
echo "$HOST:$PORT is ready"
