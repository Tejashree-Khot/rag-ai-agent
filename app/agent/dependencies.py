"""Dependency injection for singleton instances."""

from functools import lru_cache

from agent.rag_agent import OrchestrateRAGAgent, ReactRAGAgent, Retriever
from core.embedder import EmbeddingClient
from core.llm import LLMClient
from memory.milvus_manager import MilvusManager
from memory.postgres import PostgresClient


@lru_cache
def get_embedding_client() -> EmbeddingClient:
    """Get or create the singleton EmbeddingClient instance.

    :return: The singleton EmbeddingClient instance.
    """
    return EmbeddingClient()


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
    return MilvusManager()


@lru_cache
def get_retriever() -> Retriever:
    """Get or create the singleton Retriever instance.

    :return: The singleton Retriever instance.
    """
    return Retriever(milvus_manager=get_milvus_manager(), embedder=get_embedding_client())


@lru_cache
def get_react_rag_agent() -> ReactRAGAgent:
    """Get or create the singleton ReactRAGAgent instance.

    :return: The singleton ReactRAGAgent instance.
    """
    return ReactRAGAgent(llm=get_llm_client(), retriever=get_retriever())


@lru_cache
def get_orchestrate_rag_agent() -> OrchestrateRAGAgent:
    """Get or create the singleton OrchestrateRAGAgent instance.

    :return: The singleton OrchestrateRAGAgent instance.
    """
    return OrchestrateRAGAgent(
        react_rag_agent=get_react_rag_agent(), postgres_client=get_postgres_client()
    )
