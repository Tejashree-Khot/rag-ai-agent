import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from agent.dependencies import get_orchestrate_rag_agent
from agent.rag_agent import OrchestrateRAGAgent
from config.schemas import UserInput

router = APIRouter()
LOGGER = logging.getLogger("service")
LOGGER.setLevel(logging.INFO)


@router.get("/health_check", include_in_schema=False)
async def health_check():
    return JSONResponse(content={"status": "ok"}, status_code=200)


@router.post("/chat")
async def chat(
    request: UserInput,
    agent: Annotated[OrchestrateRAGAgent, Depends(get_orchestrate_rag_agent)]
) -> JSONResponse:
    session_id = request.session_id or "session-123"
    user_input = request.user_input
    try:
        result = await agent.run(session_id, user_input)
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        LOGGER.exception("OrchestrateRAGAgent failed.")
        return JSONResponse(content={"error": f"OrchestrateRAGAgent error: {e}"}, status_code=500)
