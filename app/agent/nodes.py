"""Orchestrator nodes."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from config.state import SessionState
from core.embedder import EmbeddingClient
from core.llm import LLMClient
from memory.milvus_manager import MilvusManager
from utils.helper import load_prompt
from utils.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger("nodes")
LOGGER.setLevel(logging.INFO)


def create_rag_tool(milvus_manager: MilvusManager, embedding_client: EmbeddingClient):
    @tool
    def rag_search(query: str) -> str:
        """Search the knowledge base for relevant information based on the query.

        Use this tool when you need to find information from the document knowledge base
        to answer user questions about the uploaded documents.

        Args:
            query: The search query to find relevant information.

        Returns:
            Retrieved context from the knowledge base.
        """
        LOGGER.info(f"RAG Tool: Searching for query: {query}")
        results = milvus_manager.search(query, embedding_client)
        retrieved_context = []
        for hits in results:
            for hit in hits:
                retrieved_context.append(
                    f"[Page {hit['entity']['page_number']}]: {hit['entity']['text_content']}"
                )
        context_str = "\n\n".join(retrieved_context)
        LOGGER.info(f"RAG Tool: Retrieved {len(retrieved_context)} context chunks")
        return (
            context_str if context_str else "No relevant information found in the knowledge base."
        )

    return rag_search


class BaseNode(ABC):
    @abstractmethod
    async def run(self, state: SessionState) -> SessionState:
        pass


class InputNode(BaseNode):
    async def run(self, state: SessionState) -> SessionState:
        return state


class AgentNode(BaseNode):
    """Agent node that interacts with LLM using tool calling."""

    system_prompt_template = load_prompt("system_prompt.md")

    def __init__(self, model: LLMClient, tools: list) -> None:
        self.model = model
        self.tools = tools
        self.llm_with_tools = model.model.bind_tools(tools)

    def prepare_system_prompt(self) -> str:
        return self.system_prompt_template.format()

    async def run(self, state: SessionState) -> dict[str, Any]:
        LOGGER.info("AgentNode: Processing with tool calling")
        messages = state.messages
        if not messages:
            system_msg = SystemMessage(content=self.prepare_system_prompt())
            user_msg = HumanMessage(content=state.user_input)
            messages = [system_msg, user_msg]

        response: AIMessage = await self.llm_with_tools.ainvoke(messages)
        LOGGER.info(f"AgentNode: Got response with {len(response.tool_calls)} tool calls")

        if not response.tool_calls:
            state.response = response.content

        return {"messages": [response]}


class ResponseNode(BaseNode):
    async def run(self, state: SessionState) -> dict[str, Any]:  # noqa: PLR6301
        LOGGER.info("ResponseNode: Extracting final response")
        if state.messages:
            last_message = state.messages[-1]
            if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                state.response = last_message.content
        return {"response": state.response}
