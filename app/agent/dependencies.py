"""Dependency injection for singleton instances."""

from functools import lru_cache

from agent.orchestration import Orchestrator
from core.embedder import EmbeddingClient, create_embedding_client
from core.llm import LLMClient
from memory.milvus_manager import MilvusManager, create_milvus_manager
from memory.postgres import PostgresClient


@lru_cache
def get_llm_client() -> LLMClient:
    """Get or create the singleton LLMClient instance.

    :return: The singleton LLMClient instance.
    """
    return LLMClient()


@lru_cache
def get_postgres_client() -> PostgresClient:
    """Get or create the singleton PostgresClient instance.

    :return: The singleton PostgresClient instance.
    """
    return PostgresClient()


@lru_cache
def get_milvus_manager() -> MilvusManager:
    """Get or create the singleton MilvusManager instance.

    :return: The singleton MilvusManager instance.
    """
    return create_milvus_manager()


@lru_cache
def get_embedding_client() -> EmbeddingClient:
    """Get or create the singleton EmbeddingClient instance.

    :return: The singleton EmbeddingClient instance.
    """
    return create_embedding_client()


@lru_cache
def get_orchestrator() -> Orchestrator:
    """Get or create the singleton Orchestrator instance.

    This ensures that the same Orchestrator instance (with its MemorySaver checkpointer)
    is reused across requests, allowing session state to persist.

    :return: The singleton Orchestrator instance.
    """
    return Orchestrator(
        llm_client=get_llm_client(),
        postgres_client=get_postgres_client(),
        milvus_manager=get_milvus_manager(),
        embedding_client=get_embedding_client(),
    )
