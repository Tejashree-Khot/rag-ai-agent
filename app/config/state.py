from typing import Any

from pydantic import BaseModel, Field


class SessionState(BaseModel):
    """The state of the session.

    Attributes:
        session_id: The ID of the current session.
        user_input: The user's most recent input.
        conversation_history: The list of messages that make up the chat history.
        retrieved_context: The list of retrieved context documents.
        response: Agent response.
    """

    session_id: str
    user_input: str | None = None
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_context: list[dict[str, Any]] = Field(default_factory=list)
    response: str = Field(default="")
