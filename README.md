# LangGraph RAG MCP

A Retrieval-Augmented Generation (RAG) system that serves LangGraph documentation through the Model Context Protocol (MCP).

## Overview

This project builds a documentation retrieval system that:

1. **Collects and processes LangGraph documentation** from the official website
2. **Creates a vector database** from this documentation for semantic search
3. **Exposes this knowledge** through the Model Context Protocol (MCP)
4. **Integrates with MCP-compatible hosts** like Cursor, Claude Desktop, or Windsurf

## How It Works

### 1. Documentation Collection and Processing (Context)

- **Scrapes and cleans** LangGraph documentation from multiple website URLs using BeautifulSoup
- **Splits text** into manageable chunks using RecursiveCharacterTextSplitter with tiktoken for accurate token counting
- **Embeds chunks** into vector representations using BAAI/bge-large-en-v1.5 embeddings
- **Stores vectors** in an SKLearnVectorStore for efficient retrieval

### 2. Retrieval System (Tool)

- Implements a retrieval function that finds the most relevant documentation chunks for a given query
- Integrates this function with language models like Claude to provide context-aware responses
- Returns formatted responses that include source attribution and relevant context

### 3. MCP Server Integration

- Wraps the retrieval tool in an MCP server using the MCP protocol
- Exposes the retrieval function as a tool that MCP-compatible hosts can use
- Provides access to both the retrieval system and additional resources (like the full documentation file)

## Requirements

- Python 3.10+
- Anthropic API key (for Claude models)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/langraph-rag-mcp.git
cd langraph-rag-mcp
```

2. Create and activate a virtual environment (recommended):
```bash
conda create -n mcp python=3.13
conda activate mcp
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Set up your API keys in a `.env` file:
```bash
echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
```

## Usage

### Running the Script

Execute the Jupyter Notebook to download documentation, process it, and set up the vector store, then:

```bash
python langgraph-mcp.py
```

This will:
- Download LangGraph documentation
- Save the full documentation to `llms_full.txt`
- Split the documents into chunks
- Create a vector store at `sklearn_vectorstore.parquet`
- Run a test query against the vector store
- Initialize the Anthropic model if configured

### Testing with MCP Inspector

1. Install the MCP Inspector:
```bash
npm install -g @modelcontextprotocol/inspector
```

2. Run the inspector:
```bash
npx @modelcontextprotocol/inspector
```

3. Configure the inspector with:
   - **COMMAND**: `/home/hub/anaconda3/envs/mcp/bin/python` (path to your Python interpreter - change this)
   - **ARGUMENTS**: `/home/hub/Desktop/ccode/langraph-rag-mcp/langgraph-mcp.py` (path to the script - change this)

### Configuring MCP Hosts

Create the appropriate configuration file for your MCP-compatible host:

```json
{
    "mcpServers": {
        "langgraph-mcp": {
            "command": "/home/hub/anaconda3/envs/mcp/bin/python",
            "args": [
                "/home/hub/Desktop/ccode/langraph-rag-mcp/langgraph-mcp.py"
            ],
            "env": {
                "ANTHROPIC_API_KEY": "<your-anthropic-api-key>"
            }
        }
    }
}
```

#### Cursor Configuration

For Cursor, save this configuration to `~/.cursor/mcp.json`.

## System Architecture

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  LangGraph    │     │  Vector       │     │  MCP          │
│  Documentation│────▶│  Database     │────▶│  Server       │
└───────────────┘     └───────────────┘     └───────────────┘
                                                    │
                                                    ▼
                              ┌───────────────────────────────────┐
                              │                                   │
                         ┌────┴────┐                    ┌─────────┴────────┐
                         │  Cursor  │                    │  Claude Desktop  │
                         └─────────┘                    └──────────────────┘
```

## Resources

[MCP From Scratch](https://www.notion.so/MCP-From-Scratch-1c1dd782d50180fda61ece359beef88c)