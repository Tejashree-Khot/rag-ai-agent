from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings


class EmbeddingClient:
    def __init__(self):
        self.model = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)

    def embed_query(self, query: str):
        """Embed a query string."""
        return self.model.embed_query(query)
