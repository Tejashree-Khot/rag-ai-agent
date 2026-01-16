"""Query Milvus."""

import sys
from pathlib import Path

from dotenv import load_dotenv

APP_PATH = Path(__file__).parent.parent / "app"
load_dotenv(APP_PATH / ".env")
sys.path.append(str(APP_PATH))

from core.embedder import EmbeddingClient
from memory.milvus_manager import MilvusManager


def execute_similarity_search(
    query: str, embedding_client: EmbeddingClient, milvus_manager: MilvusManager
):
    """Execute similarity search."""
    results = milvus_manager.search(query, embedding_client)
    for search_hits in results:
        for search_hit in search_hits:
            score = search_hit["distance"]
            page_number = search_hit["entity"]["page_number"]
            content = search_hit["entity"]["text_content"][:150].replace("\n", " ")
            print(f"[Score: {score:.4f} | Page: {page_number}] - Text: {content}...")


def main():
    embedding_client = EmbeddingClient()
    milvus_manager = MilvusManager()

    query = "What is the conclusion of the paper?"
    print(f"Testing Search - '{query}'")
    execute_similarity_search(query, embedding_client, milvus_manager)


if __name__ == "__main__":
    main()
