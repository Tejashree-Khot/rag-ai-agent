from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import settings


class LLMClient:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL_NAME,
            google_api_key=settings.GEMINI_API_KEY.get_secret_value(),
            temperature=0.2,
        )

    async def ainvoke(self, messages: list[BaseMessage]):
        """Invoke the LLM with messages.

        Args:
            messages: Either a string prompt or a list of message dicts

        Returns:
            The response content as a string
        """
        response = await self.model.ainvoke(messages)
        return response.content
