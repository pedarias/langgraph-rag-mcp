# LangGraph RAG MCP

A Retrieval-Augmented Generation (RAG) system that serves LangGraph documentation through the Model Context Protocol (MCP).

## Overview

This project builds a documentation retrieval system that:

1. **Collects and processes LangGraph documentation** from the official website
2. **Creates a vector database** from this documentation for semantic search
3. **Exposes this knowledge** through the Model Context Protocol (MCP)
4. **Integrates with MCP-compatible hosts** like VS Code, Cursor, Claude Desktop, or Windsurf

## How It Works

### 1. Documentation Collection and Processing (Context)

- **Recursively scrapes and cleans** LangGraph documentation from multiple website URLs using `RecursiveUrlLoader` and BeautifulSoup.
- **Splits text** into manageable chunks using `RecursiveCharacterTextSplitter` with `tiktoken` for accurate token counting.
- **Embeds chunks** into vector representations using `BAAI/bge-large-en-v1.5` embeddings.
- **Stores vectors** in an `SKLearnVectorStore` for efficient retrieval.

### 2. Retrieval System (Tool)

- Implements a retrieval function that finds the most relevant documentation chunks for a given query.
- Integrates this function with language models like Claude to provide context-aware responses.
- Returns formatted responses that include source attribution and relevant context.

### 3. MCP Server Integration

- Wraps the retrieval tool in an MCP server using the `fastmcp` library.
- Exposes the retrieval function as a tool that MCP-compatible hosts can use.
- Provides access to both the retrieval system and additional resources (like the full documentation file).

## Requirements

- Python 3.10+
- Docker and Docker Compose (recommended)
- Anthropic API key (for Claude models)

## Installation and Setup

You can run this project either with Docker (recommended) or in a local Python environment.

### Using Docker (Recommended)

1. **Clone this repository:**
    ```bash
    git clone https://github.com/yourusername/langraph-rag-mcp.git
    cd langraph-rag-mcp
    ```

2. **Set up your API keys in a `.env` file:**
    Create a `.env` file in the project root and add your Anthropic API key:
    ```bash
    echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
    ```
    The `docker-compose.yml` file will automatically load this environment variable.

### Local Environment (without Docker)

1. **Clone this repository:**
    ```bash
    git clone https://github.com/yourusername/langraph-rag-mcp.git
    cd langraph-rag-mcp
    ```

2. **Create and activate a virtual environment:**
    ```bash
    conda create -n mcp python=3.13
    conda activate mcp
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up your API keys in a `.env` file:**
    ```bash
    echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
    ```

## Usage

The process involves two main steps: first, generating the vector store, and second, running the MCP server.

### Step 1: Generate the Vector Store

You only need to do this once, or whenever you want to update the documentation.

1.  Open and run the `rag-tool.ipynb` notebook in a Jupyter environment.
2.  This will:
    -   Download the latest LangGraph documentation.
    -   Save the full documentation to `llms_full.txt`.
    -   Split the documents into chunks.
    -   Create and persist a vector store at `sklearn_vectorstore.parquet`.

### Step 2: Run the MCP Server

#### With Docker

The easiest way to run the server is using the provided shell script, which wraps Docker Compose.

```bash
bash run-mcp-docker.sh
```

This script will build the Docker image if it doesn't exist, start the container, and then execute the MCP server inside it, correctly handling standard I/O for MCP communication.

#### Without Docker

If you are not using Docker, you can run the MCP server directly in dev mode through the command:

```bash
mcp dev langgraph-mcp.py
```

## Configuring MCP Hosts

To use this MCP server with a compatible editor, you need to configure it.

### VS Code

1.  Open your VS Code `settings.json` file. (You can find it via the command palette: `Preferences: Open User Settings (JSON)`).
2.  Add the following configuration to the file. Make sure to replace `<path-to-your-project>` with the absolute path to the `langraph-rag-mcp` directory on your machine.

```json
"mcp.servers": [
    {
        "name": "langgraph-mcp",
        "command": [
            "/bin/bash",
            "<path-to-your-project>/run-mcp-docker.sh"
        ],
        "env": {
            "ANTHROPIC_API_KEY": "<your-anthropic-api-key>"
        }
    }
]
```

If you are not using Docker, change the `command` to:
```json
"command": [
    "<path-to-your-conda-env>/bin/python",
    "<path-to-your-project>/langgraph-mcp.py"
]
```

## System Architecture

```
(Phase 1: Data Ingestion - Performed once on Host Machine)
┌─────────────┐      ┌──────────────────┐      ┌───────────────────────────────┐
│ LangGraph   │      │ Jupyter Notebook │      │ Vector Store & Full Docs      │
│ Docs (Web)  │───▶  │ (rag-tool.ipynb) │───▶  │ (.parquet & .txt files)       │
└─────────────┘      └──────────────────┘      └───────────────────────────────┘


(Phase 2: Live RAG System - Request/Response Flow)
┌──────────────┐                             ┌─────────────┐
│   VS Code    │                             │  .env file  │
│(User Interface)│                             │ (API KEY)   │
└──────┬───────┘                             └──────┬──────┘
       │ 1. User Query                             │ (provides)
       ▼                                           │
┌──────┴───────────────────────────────────────────┴───────────────────────────────┐
│ Host Machine Boundary                                                            │
│                                                                                  │
│      ┌────────────────────┐          ┌─────────────────────────────────────────┐ │
│      │ run-mcp-docker.sh  │ 2. Execs │  🐳 Docker Container                      │ │
│      │ (Entrypoint Script)│────▶     │                                         │ │
│      └────────────────────┘          │  ┌───────────────────────────────────┐  │ │
│                                      │  │   🐍 Python MCP Server              │  │ │
│      ▲                             ◀─┼──│   (langgraph-mcp.py)              │  │ │
│      │ 7. Final Response             │  └───────────────┬───────────────────┘  │ │
│      │                               │                  │ 3. Reads Data From  │ │
│      └───────────────────────────────│──────────────────│─────────────────────┘ │
│                                      │                  │                       │
│                                      │                  ▼                       │
│                                      │  ┌───────────────────────────────────┐  │
│ (files mounted from Host) ············  │   Mounted Vector Store & Docs     │  │
│                                      │  └───────────────────────────────────┘  │
│                                      └─────────────────────────────────────────┘ │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                                        │ 4. API Call
                                                        │    (sends augmented prompt)
                                                        ▼
                                             ┌────────────────────┐
                                             │ ☁️ Anthropic API   │
                                             │   (Claude LLM)     │
                                             └──────────┬─────────┘
                                                        │ 5. Generation
                                                        │
                                                        ◀························
                                                          6. Returns Response
                                                          (to MCP Server)
```

## Resources

[MCP From Scratch](https://www.notion.so/MCP-From-Scratch-1c1dd782d50180fda61ece359beef88c)