import os
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.core.llm import embeddings
from src.config.settings import settings
from src.utils.data_loader import load_filtered_docs_from_json

class VectorStoreManager:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name="isro_collection",
            embedding_function=embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
        )

    def ingest_data(self):
        """
        Loads data from JSON, splits it, and adds it to the vector store.
        """
        if not os.path.exists(settings.DATA_FILE_PATH):
            print(f"Data file not found at {settings.DATA_FILE_PATH}. Skipping ingestion.")
            return

        documents = load_filtered_docs_from_json(settings.DATA_FILE_PATH)
        if not documents:
            print("No documents loaded.")
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(documents)
        
        self.vector_store.add_documents(documents=docs)
        print(f"Ingested {len(docs)} chunks into vector store.")

    def get_retriever(self, k=2):
        return self.vector_store.as_retriever(search_kwargs={"k": k})
    
    def similarity_search(self, query, k=2):
        return self.vector_store.similarity_search(query, k=k)

vector_store_manager = VectorStoreManager()
