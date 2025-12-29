import logging

from pymilvus import DataType, MilvusClient

from config.settings import settings
from core.embedder import EmbeddingClient
from utils.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger("milvus_manager")
LOGGER.setLevel(logging.INFO)


class MilvusManager:
    def __init__(self):
        self.client = MilvusClient(uri=settings.MILVUS_URI, token=settings.MILVUS_TOKEN)
        self.batch_size = settings.BATCH_SIZE
        self.collection_name = settings.MILVUS_COLLECTION_NAME
        self.embedding_dim = settings.EMBEDDING_DIM
        LOGGER.info(f"Connected to Milvus at {settings.MILVUS_URI}")

    def create_collection(self, drop_old=True):
        """Defines schema and creates the collection."""
        if drop_old and self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            LOGGER.info(f"Dropped existing collection: {self.collection_name}")

        LOGGER.info("Creating Schema")
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)

        # Schema Definition
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        schema.add_field("text_content", DataType.VARCHAR, max_length=65535)
        schema.add_field("page_number", DataType.INT64)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector", metric_type="COSINE", index_type="IVF_FLAT", params={"nlist": 128}
        )

        self.client.create_collection(
            collection_name=self.collection_name, schema=schema, index_params=index_params
        )
        LOGGER.info(f"Collection '{self.collection_name}' created successfully.")

    def insert_chunks(self, chunks, embedding_client):
        """Embeds text chunks and inserts them into Milvus in batches."""
        data_rows = []
        total_chunks = len(chunks)

        LOGGER.info("Starting Insertion")
        for i, chunk in enumerate(chunks):
            text = chunk.page_content
            if not text.strip():
                continue

            # Generate Vector
            vector = embedding_client.embed_query(text)
            page_num = chunk.metadata.get("page", 0)

            data_rows.append({"vector": vector, "text_content": text, "page_number": page_num})

            # Batch Insert
            if len(data_rows) >= self.batch_size:
                self.client.insert(self.collection_name, data_rows)
                LOGGER.info(f"Inserted batch {i + 1}/{total_chunks}")
                data_rows = []

        # Insert remaining
        if data_rows:
            self.client.insert(self.collection_name, data_rows)
            LOGGER.info("Inserted final batch.")

    def search(self, query_text: str, embedding_model: EmbeddingClient, limit: int = 3):
        """Performs a semantic search."""
        query_vector = embedding_model.embed_query(query_text)

        return self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=["text_content", "page_number"],
        )


def create_milvus_manager() -> MilvusManager:
    milvus_manager = MilvusManager()
    # Only create collection if it doesn't exist
    if not milvus_manager.client.has_collection(milvus_manager.collection_name):
        LOGGER.info(f"Collection '{milvus_manager.collection_name}' does not exist. Creating...")
        milvus_manager.create_collection(drop_old=True)
    return milvus_manager
