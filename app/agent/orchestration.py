"""Agent Orchestrator logic."""

import logging

from langchain_core.runnables import RunnableConfig

from agent.graph_builder import GraphBuilder
from agent.nodes import AgentNode, InputNode, ResponseNode, create_rag_tool
from config.state import SessionState
from core.embedder import EmbeddingClient
from core.llm import LLMClient
from memory.milvus_manager import MilvusManager
from memory.postgres import PostgresClient
from utils.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger("agent")
LOGGER.setLevel(logging.INFO)


class Nodes:
    """Container for all orchestration nodes."""

    def __init__(self, llm_client: LLMClient, tools: list) -> None:
        self.input_guardrail = InputNode()
        self.response = ResponseNode()
        self.agent_node = AgentNode(llm_client, tools)


class Orchestrator:
    def __init__(
        self,
        llm_client: LLMClient,
        postgres_client: PostgresClient,
        milvus_manager: MilvusManager,
        embedding_client: EmbeddingClient,
    ):
        self.llm_client = llm_client
        self.postgres_client = postgres_client
        self.milvus_manager = milvus_manager
        self.embedding_client = embedding_client

        rag_tool = create_rag_tool(milvus_manager, embedding_client)
        self.tools = [rag_tool]
        self.nodes = Nodes(llm_client, self.tools)

        self.graph_builder = GraphBuilder(self)
        self.graph = self.graph_builder.build()

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
        """Run the orchestrator."""
        LOGGER.info("Orchestrator started.")
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        state = await self.load_state_memory(session_id)
        LOGGER.info("Loaded state memory")
        state.user_input = user_input

        try:
            state_dict = await self.graph.ainvoke(state.model_dump(), config, stream_mode="values")
            LOGGER.info("Orchestrator completed.")

            state_from_result = SessionState(**state_dict)
            await self.save_state_memory(state_from_result)

            return state_from_result.model_dump()

        except Exception as e:
            LOGGER.exception("Orchestrator failed.")
            raise RuntimeError(f"Orchestrator failed: {e}") from e


if __name__ == "__main__":
    orchestrator = Orchestrator(LLMClient(), PostgresClient(), MilvusManager(), EmbeddingClient())
    graph = orchestrator.graph
    mermaid_code = graph.get_graph().draw_mermaid()
    print(mermaid_code)
