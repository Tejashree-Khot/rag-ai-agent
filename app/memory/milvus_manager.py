import logging

from pymilvus import MilvusClient

from config.settings import settings
from core.embedder import EmbeddingClient
from utils.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger("milvus_manager")
LOGGER.setLevel(logging.INFO)


class MilvusManager:
    def __init__(self):
        self.client = MilvusClient(uri=settings.MILVUS_URI, token=settings.MILVUS_TOKEN)
        self.collection_name = settings.MILVUS_COLLECTION_NAME
        LOGGER.info(f"Connected to Milvus at {settings.MILVUS_URI}")

    def search(self, query_text: str, embedding_client: EmbeddingClient, limit: int = 3):
        """Performs a semantic search."""
        query_vector = embedding_client.embed_query(query_text)
        return self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=["text_content", "page_number"],
        )
