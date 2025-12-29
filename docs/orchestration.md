# LangGraph Orchestration Flow

## Orchestration Graph

```mermaid
---
config:
  flowchart:
    curve: curve
---
%%{init: {'theme': 'neutral'}}%%
graph TD
    START([START]) --> input_guard
    
    %% Input Guard analyzes safety
    input_guard{Input Safe?} 
    input_guard -- Yes --> agent
    input_guard -- No --> END([END])

    %% Agent Decision Loop
    agent{Agent Router}
    agent -- Call RAG --> rag_tool
    rag_tool -- Return Context --> agent
    
    %% Final Generation
    agent -- Generate Answer --> response_generator
    response_generator --> END

    %% Styling
    classDef processNode fill:#e1f5fe,stroke:#01579b,stroke-width:2px,rx:10,ry:10
    classDef decisionNode fill:#fff3e0,stroke:#e65100,stroke-width:2px,rx:10,ry:10
    classDef toolNode fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,rx:10,ry:10
    classDef startEndNode fill:#c8e6c9,stroke:#1b5e20,stroke-width:3px,rx:10,ry:10

    class rag_tool toolNode
    class agent,response_generator processNode
    class input_guard decisionNode
    class START,END startEndNode

```

## Flow Description

### Phase 1: Input Analysis

* **START** → `input_guard`: The entry point where the user query is received.
* `input_guard`: checks the query for safety (PII, jailbreaks, etc.) and appropriateness. If unsafe, it terminates the flow immediately.

### Phase 2: Agent Reasoning

* **Agent Loop**: The `agent` acts as the reasoning engine (Router). It determines if it has enough information to answer:
* If **No**: It routes to the `rag_tool`.
* If **Yes**: It routes to the `response_generator`.

### Phase 3: Retrieval (RAG)

* `rag_tool`: Retrieves relevant documents from the vector database (e.g., Milvus) based on the query.
* **Return**: The retrieved context is passed back to the `agent` to re-evaluate the answer.

### Phase 4: Response Generation

* `response_generator`: Synthesizes the final answer using the chat history and retrieved context, formatting it strictly for the user.
* **END**: Delivers the final payload to the client.

### Nodes Dictionary

| Node Name | Type | Description |
| --- | --- | --- |
| `input_guard` | Conditional | Validates input for safety. Redirects unsafe inputs to END. |
| `agent` | LLM / Router | The central brain. Decides whether to call tools or generate a final answer. |
| `rag_tool` | Tool | Performs semantic search to retrieve external context. |
| `response_generator` | Output | Formats the final answer into the desired schema/style. |