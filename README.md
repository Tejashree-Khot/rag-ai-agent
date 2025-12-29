# RAG AI Agent

AI Agent with integrated RAG tool calls

## Setup

```bash
uv venv
uv sync --all-groups --frozen --active
```

## Running the Application

To start the FastAPI server locally:
In terminal 1

```bash
cd app
docker-compose up
```

API will be available at `http://localhost:8080`.

In terminal 2

```bash
uv run streamlit run scripts/chatbot.py
```

The UI will be available at `http://localhost:8501`

[Agent Orchestration Flow](docs/orchestration.md)
