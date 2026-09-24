#!/bin/bash

# Docker wrapper script for MCP server
# This script handles the stdio communication between VS Code and the containerized MCP server

set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Resolve Compose independently of the MCP host's working directory
COMPOSE_FILE="${PROJECT_DIR}/docker-compose.yml"

# Build explicitly before use so build output never enters the MCP stream
if [[ "${1:-}" == "--build" && "$#" -eq 1 ]]; then
    docker compose --project-directory "${PROJECT_DIR}" -f "${COMPOSE_FILE}" build >&2
elif [[ "$#" -gt 0 ]]; then
    echo "Usage: $0 [--build]" >&2
    exit 2
fi
IMAGE="$(docker compose --project-directory "${PROJECT_DIR}" -f "${COMPOSE_FILE}" config --images)"
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "Build the image first with docker compose build, or run this script with --build." >&2
    exit 1
fi

# Execute the MCP server inside a disposable container with stdio
exec docker compose --project-directory "${PROJECT_DIR}" -f "${COMPOSE_FILE}" \
    run --rm --no-deps -T langgraph-mcp serve