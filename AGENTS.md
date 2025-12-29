# Repo description

RAG AI Agent with integrated tool calling built on LangGraph. The agent uses semantic search to retrieve relevant context from uploaded documents and generate informed responses.

## Repo structure

```text
rag-ai-agent/
├── app/                # Main application package
│   ├── agent/          # LangGraph orchestration
│   ├── config/         # Configuration and state definitions
│   ├── core/           # LLM and embedding clients
│   ├── memory/         # Milvus and PostgreSQL persistence
│   ├── prompts/        # Prompt templates
│   ├── service/        # FastAPI routes and utilities
│   └── utils/          # Logging and PDF processing
├── docs/               # Documentation
└── scripts/            # Utility scripts (Streamlit UI)
```

## Development Rules

1. Always use `uv run` to execute Python scripts.
2. Do **not** add test cases unless explicitly requested.
3. Do **not** add comments in the code. Docstrings are allowed.
4. Do **not** over-engineer the solution.
5. Use **classes and functions** wherever appropriate.
6. Follow the **existing project structure and coding style** when adding new code.
7. **Always confirm with me before changing or introducing a design pattern.**
8. Implement **only the requested functionality** — do not add extras.
9. Add code in **logical order**; if no clear order exists, use **alphabetical order**.

## Coding practices

- **Type hints**: Use type annotations for function signatures and class attributes
- **Pydantic**: Use `BaseModel` for data validation and `BaseSettings` for configuration
- **Async/await**: Use async patterns for I/O operations (database, LLM calls)
- **Dependency injection**: Use `@lru_cache` decorated functions for singleton instances
