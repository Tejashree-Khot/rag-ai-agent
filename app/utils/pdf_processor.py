import logging
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from utils.logger import configure_logging

configure_logging()
LOGGER = logging.getLogger("pdf_processor")
LOGGER.setLevel(logging.INFO)


class PDFProcessor:
    def __init__(self, file_path: Path):
        self.file_path = file_path

    @staticmethod
    def load_and_chunk(file_path: Path) -> list[Document]:
        """Loads PDF and splits it into chunks."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find {file_path}")

        LOGGER.info(f"Loading PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        LOGGER.info(f"Loaded {len(docs)} pages.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = text_splitter.split_documents(docs)
        LOGGER.info(f"Split PDF into {len(chunks)} text chunks.")
        return chunks
