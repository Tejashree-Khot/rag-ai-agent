import json
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from config.settings import settings
from config.state import SessionState


def get_postgres_connection_string() -> str:
    """Build and return the PostgreSQL connection string from settings."""
    if settings.POSTGRES_PASSWORD is None:
        raise ValueError("POSTGRES_PASSWORD is not set")
    return (
        f"postgresql://{settings.POSTGRES_USER}:"
        f"{settings.POSTGRES_PASSWORD.get_secret_value()}@"
        f"{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/"
        f"{settings.POSTGRES_DB}"
    )


@asynccontextmanager
async def get_postgres_saver():
    "Initializes and return a postgreSQL saver instance using connection pool for resilent connection"

    application_name = settings.POSTGRES_APPLICATION_NAME + "-" + "saver"

    async with AsyncConnectionPool(
        get_postgres_connection_string(),
        min_size=settings.POSTGRES_MIN_CONNECTIONS_PER_POOL,
        max_size=settings.POSTGRES_MAX_CONNECTIONS_PER_POOL,
        kwargs={"autocommit": True, "row_factory": dict_row, "application_name": application_name},
        check=AsyncConnectionPool.check_connection,
    ) as pool:
        try:
            async with pool.connection() as conn:
                checkpointer = AsyncPostgresSaver(conn)  # type: ignore
                await checkpointer.setup()
                yield checkpointer

        finally:
            await pool.close()


@asynccontextmanager
async def get_postgres_store():
    "Initializes and return a postgreSQL store instance using connection pool for resilent connection"

    application_name = settings.POSTGRES_APPLICATION_NAME + "-" + "store"

    async with AsyncConnectionPool(
        get_postgres_connection_string(),
        min_size=settings.POSTGRES_MIN_CONNECTIONS_PER_POOL,
        max_size=settings.POSTGRES_MAX_CONNECTIONS_PER_POOL,
        kwargs={"autocommit": True, "row_factory": dict_row, "application_name": application_name},
        check=AsyncConnectionPool.check_connection,
    ) as pool:
        try:
            async with pool.connection() as conn:
                store = AsyncPostgresStore(conn)  # type: ignore
                await store.setup()
                yield store

        finally:
            await pool.close()


class PostgresClient:
    def __init__(self):
        self.connection_string = get_postgres_connection_string()
        self.pool = None

    async def ensure_pool(self):
        if self.pool is None:
            self.pool = AsyncConnectionPool(
                self.connection_string,
                min_size=settings.POSTGRES_MIN_CONNECTIONS_PER_POOL,
                max_size=settings.POSTGRES_MAX_CONNECTIONS_PER_POOL,
                kwargs={
                    "autocommit": True,
                    "row_factory": dict_row,
                    "application_name": settings.POSTGRES_APPLICATION_NAME,
                },
                check=AsyncConnectionPool.check_connection,
                open=False,
            )
            await self.pool.open()

    async def create_tables(self):
        await self.ensure_pool()
        async with self.pool.connection() as conn:  # type: ignore
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS session_state (
                    session_id TEXT PRIMARY KEY,
                    user_input TEXT,
                    conversation_history JSONB,
                    retrieved_context JSONB,
                    response TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

    async def add_state(self, state: SessionState):
        await self.ensure_pool()
        async with self.pool.connection() as conn:  # type: ignore
            await conn.execute(
                """
                INSERT INTO session_state (
                    session_id, 
                    user_input, 
                    conversation_history,
                    retrieved_context,
                    response
                )
                VALUES (
                    %(session_id)s, 
                    %(user_input)s, 
                    %(conversation_history)s,
                    %(retrieved_context)s,
                    %(response)s
                )
                ON CONFLICT (session_id) DO UPDATE SET
                    user_input = EXCLUDED.user_input,
                    conversation_history = EXCLUDED.conversation_history,
                    retrieved_context = EXCLUDED.retrieved_context,
                    response = EXCLUDED.response,
                    updated_at = CURRENT_TIMESTAMP;
                """,
                {
                    "session_id": state.session_id,
                    "user_input": state.user_input,
                    "conversation_history": json.dumps(state.conversation_history),
                    "retrieved_context": json.dumps(state.retrieved_context),
                    "response": state.response,
                },
            )

    async def get_state(self, session_id: str) -> SessionState | None:
        await self.ensure_pool()
        async with self.pool.connection() as conn:  # type: ignore
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT * FROM session_state WHERE session_id = %(session_id)s",
                    {"session_id": session_id},
                )
                row = await cur.fetchone()
                if row:
                    parsed_row = dict(row)
                    if "conversation_history" in parsed_row and isinstance(
                        parsed_row["conversation_history"], str
                    ):
                        parsed_row["conversation_history"] = json.loads(
                            parsed_row["conversation_history"]
                        )
                    if "retrieved_context" in parsed_row and isinstance(
                        parsed_row["retrieved_context"], str
                    ):
                        parsed_row["retrieved_context"] = json.loads(
                            parsed_row["retrieved_context"]
                        )
                    return SessionState(**parsed_row)
        return None

    async def close(self):
        if self.pool:
            await self.pool.close()
