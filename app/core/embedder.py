from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config.settings import settings


class EmbeddingClient:
    def __init__(self):
        self.model = GoogleGenerativeAIEmbeddings(  # type: ignore[call-arg]
            model=settings.EMBEDDING_MODEL_NAME,
            google_api_key=settings.GEMINI_API_KEY.get_secret_value(),
        )

    def embed_query(self, query: str) -> list[float]:
        """Embed a query string."""
        return self.model.embed_query(query)
