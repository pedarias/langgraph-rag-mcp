#!/bin/bash

# Docker wrapper script for MCP server
# This script handles the stdio communication between VS Code and the containerized MCP server

CONTAINER_NAME="langgraph-mcp-container"

# Ensure the container is up and running with the latest build
echo "Starting/Updating LangGraph MCP container..." >&2
docker compose up --build -d

# Wait for container to be ready
sleep 2

# Execute the MCP server inside the container with stdio
exec docker exec -i "${CONTAINER_NAME}" python langgraph-mcp.py