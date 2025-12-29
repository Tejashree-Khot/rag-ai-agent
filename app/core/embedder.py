from abc import ABC, abstractmethod

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings


class EmbeddingClient(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def embed_documents(self, documents: list[str]):
        """Embed a list of text strings."""
        pass

    @abstractmethod
    def embed_query(self, query: str):
        """Embed a query string."""
        pass


class HuggingFaceEmbeddingClient(EmbeddingClient):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = HuggingFaceEmbeddings(model_name=self.model_name)

    def embed_documents(self, documents: list[str]):
        """Embed a list of text strings."""
        return self.model.embed_documents(documents)

    def embed_query(self, query: str):
        """Embed a query string."""
        return self.model.embed_query(query)


def create_embedding_client() -> EmbeddingClient:
    if settings.EMBEDDING_PROVIDER == "huggingface":
        return HuggingFaceEmbeddingClient(model_name=settings.EMBEDDING_MODEL_NAME)
    else:
        raise ValueError("Unsupported embedding provider")
