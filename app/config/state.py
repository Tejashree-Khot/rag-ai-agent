from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class SessionState(BaseModel):
    """The state of the graph.

    Attributes:
        session_id: The ID of the current user.
        user_input: The user's most recent user_input.
        messages: The list of messages for LangGraph tool calling.
        conversation_history: The list of messages that make up the chat history.
        retrieved_context: The list of retrieved context documents.
        response: Agent response.
    """

    session_id: str
    user_input: str | None = None

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_context: list[dict[str, Any]] = Field(default_factory=list)
    response: str = Field(default="")
