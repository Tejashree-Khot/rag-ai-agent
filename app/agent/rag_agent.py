import json
import logging
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from config.state import SessionState
from core.embedder import EmbeddingClient
from core.llm import LLMClient
from memory.milvus_manager import MilvusManager
from memory.postgres import PostgresClient
from utils.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger("rag_agent")
LOGGER.setLevel(logging.INFO)


class Retriever:
    """Milvus cosine retriever."""

    def __init__(
        self, milvus_manager: MilvusManager, embedder: EmbeddingClient = None, top_k: int = 5
    ):
        self.embedder = embedder
        self.top_k = top_k
        self.milvus_manager = milvus_manager

    def retrieve(self, query: str) -> list[str]:
        """Retrieve relevant documents for a query."""
        results = self.milvus_manager.search(query, self.embedder, limit=self.top_k)
        contexts = []
        for search_hits in results:
            for search_hit in search_hits:
                contexts.append(search_hit["entity"]["text_content"])
        return contexts


class RetrieveContextInput(BaseModel):
    question: str = Field(description="The question to retrieve context for")


def build_retrieval_tool(retriever: Retriever) -> StructuredTool:
    """Build a retrieval tool."""

    def retrieve_fn(question: str) -> str:
        """Retrieve relevant documents for a question."""
        LOGGER.info(f"Tool called with question: {question}")
        retrieved_contexts = retriever.retrieve(question)
        LOGGER.info(f"Retrieved {len(retrieved_contexts)} contexts")
        result = json.dumps({"retrieved_contexts": retrieved_contexts}, ensure_ascii=False)
        return result

    return StructuredTool(
        name="retrieve_context",
        description="Retrieve relevant context passages for a question and return the chunk ids",
        func=retrieve_fn,
        args_schema=RetrieveContextInput,
    )


class ReactRAGAgent:
    """Simple LangChain ReAct RAG Agent."""

    system_prompt = (
        "You are a RAG agent. "
        "Analyze the user's question and determine if it is a question about the thesis or a general question."
        "If it is a general question, answer it based on your knowledge. "
        "If it is a question about the AI, LLM, RAG, or any other AI-related topic, use the `retrieve_context` tool to fetch relevant information."
        "If the knowledge base returns no relevant results, acknowledge this and provide general assistance."
    )

    def __init__(self, llm: LLMClient, retriever: Retriever):
        self.retriever = retriever
        self.tool = build_retrieval_tool(retriever)
        self.agent: Any = create_agent(
            model=llm.model, tools=[self.tool], system_prompt=self.system_prompt
        )

    def update_state(self, state: SessionState, user_input: str, result: dict) -> SessionState:
        """Update session state from agent result."""
        state.user_input = user_input
        state.response = result["response"]
        retrieved_contexts = result.get("retrieved_contexts", [])
        state.retrieved_context = [
            {"content": context} if isinstance(context, str) else context
            for context in retrieved_contexts
        ]
        state.conversation_history.append({"role": "user", "content": user_input})
        state.conversation_history.append({"role": "assistant", "content": result["response"]})
        return state

    def prepare_messages(self, state: SessionState, current_prompt: str) -> list[dict[str, Any]]:
        """Build LLM-formatted messages list from conversation history and current prompt."""
        messages = []

        if state.conversation_history:
            for message in state.conversation_history:
                if message["role"] == "user":
                    messages.append(HumanMessage(content=message["content"]))
                elif message["role"] == "assistant":
                    messages.append(AIMessage(content=message["content"]))
                else:
                    raise ValueError(f"Invalid message role: {message['role']}")

        messages.append(HumanMessage(content=current_prompt))
        return messages

    async def ainvoke(self, state: SessionState) -> SessionState:
        """Invoke the agent."""
        if not state.user_input:
            state.response = "Please provide a question."
            return state

        messages = self.prepare_messages(state, state.user_input)
        result = await self.agent.ainvoke({"messages": messages})

        messages = result.get("messages", [])
        LOGGER.info(f"Agent returned {len(messages)} messages")
        response = ""
        retrieved_contexts: list[dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_output = json.loads(msg.content)  # type: ignore
                retrieved_contexts = tool_output.get("retrieved_contexts", [])
                LOGGER.info(f"Parsed ToolMessage: {len(retrieved_contexts)} contexts")
            elif isinstance(msg, AIMessage):
                response = str(msg.content)

        result = {"response": response, "retrieved_contexts": retrieved_contexts}
        state = self.update_state(state, state.user_input, result)
        return state


class OrchestrateRAGAgent:
    """Orchestrate the RAG agent."""

    def __init__(self, react_rag_agent: ReactRAGAgent, postgres_client: PostgresClient):
        self.react_rag_agent = react_rag_agent
        self.postgres_client = postgres_client

    async def load_state_memory(self, session_id: str) -> SessionState:
        """Load state memory from Postgres."""
        await self.postgres_client.create_tables()
        state = await self.postgres_client.get_state(session_id)
        if state:
            return state
        return SessionState(session_id=session_id)

    async def save_state_memory(self, state: SessionState) -> None:
        """Save state memory to Postgres."""
        await self.postgres_client.add_state(state)

    async def run(self, session_id: str, user_input: str) -> dict:
        """Run the agent."""
        state = await self.load_state_memory(session_id)
        state.user_input = user_input
        try:
            state = await self.react_rag_agent.ainvoke(state)
            await self.save_state_memory(state)
            return state.model_dump()
        except Exception as e:
            LOGGER.exception("ReactRAGAgent failed.")
            raise RuntimeError(f"ReactRAGAgent failed: {e}") from e
