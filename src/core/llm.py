from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from src.config.settings import settings

class LLMFactory:
    @staticmethod
    def get_llm():
        return ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            google_api_key=settings.GOOGLE_API_KEY
        )

    @staticmethod
    def get_embeddings():
        return GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY
        )

llm = LLMFactory.get_llm()
embeddings = LLMFactory.get_embeddings()
