"""
MCP Server for OpenMemory with resilient memory client handling.

This module implements an MCP (Model Context Protocol) server that provides
memory operations for OpenMemory. The memory client is initialized lazily
to prevent server crashes when external dependencies (like Ollama) are
unavailable. If the memory client cannot be initialized, the server will
continue running with limited functionality and appropriate error messages.

Key features:
- Lazy memory client initialization
- Graceful error handling for unavailable dependencies
- Fallback to database-only mode when vector store is unavailable
- Proper logging for debugging connection issues
- Environment variable parsing for API keys
"""

import logging
import asyncio
import json
from mcp.server.fastmcp import FastMCP
import time
from mcp.server.sse import SseServerTransport
from app.utils.memory import get_memory_client
from fastapi import FastAPI, Request
from fastapi.routing import APIRouter
import contextvars
import os
from dotenv import load_dotenv
from app.database import SessionLocal
from app.models import Memory, MemoryState, MemoryStatusHistory, MemoryAccessLog
from app.utils.db import get_user_and_app
import uuid
import datetime
from app.utils.permissions import check_memory_access_permissions
from qdrant_client import models as qdrant_models
from typing import Any, Dict, Optional

# Load environment variables
load_dotenv()

# Set up enhanced logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.debug("MCP Server starting up...")
logger.info("MCP Server module loaded")

# Initialize MCP
mcp = FastMCP("mem0-mcp-server")
logger.debug("FastMCP initialized")

# Don't initialize memory client at import time - do it lazily when needed
def get_memory_client_safe():
    """Get memory client with error handling. Returns None if client cannot be initialized."""
    try:
        return get_memory_client()
    except Exception as e:
        logging.warning(f"Failed to get memory client: {e}")
        return None

# Context variables for user_id and client_name
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id")
client_name_var: contextvars.ContextVar[str] = contextvars.ContextVar("client_name")

# Create a router for MCP endpoints
mcp_router = APIRouter(prefix="/mcp")

# Initialize SSE transport
sse = SseServerTransport("/mcp/messages/")

# Helper function to extract user_id from metadata
def get_user_id_from_metadata(metadata):
    """
    Extract user_id from metadata's conversationId.
    Returns: user_id string or None if not found
    """
    if isinstance(metadata, dict) and "conversationId" in metadata:
        user_id = metadata["assistantId"]
        logging.info(f"Extracted user_id from conversationId: {user_id}")
        return user_id
    return None

async def _process_add_memories(
    text: str,
    full_meta: Dict[str, Any],
    agent_id: Optional[str],
    user_id: str,
    client_name: str,
    original_metadata: Dict[str, Any],
):
    """Background task to perform add_memories work without blocking SSE."""
    try:
        memory_client = get_memory_client_safe()
        if not memory_client:
            logging.error("Background add_memories: memory client unavailable")
            return

        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=user_id, app_id=client_name)
            if not app.is_active:
                logging.warning(f"Background add_memories: app {app.name} is paused; skipping")
                return

            add_params: Dict[str, Any] = {
                "user_id": user_id,
                "metadata": full_meta,
            }
            if agent_id:
                add_params["agent_id"] = agent_id

            logging.debug("Starting memory_client.add call (background)")
            add_start = time.perf_counter()
            try:
                response = await asyncio.to_thread(memory_client.add, text, **add_params)
            finally:
                logging.debug(
                    f"memory_client.add (background) completed in {time.perf_counter() - add_start:.3f}s"
                )

            logging.debug(f"memory_client.add (background) response: {response}")

            if isinstance(response, dict) and 'results' in response:
                for result in response['results']:
                    mem_id = uuid.UUID(result['id'])
                    memory = db.query(Memory).filter(Memory.id == mem_id).first()
                    if result['event'] == 'ADD':
                        if not memory:
                            memory = Memory(
                                id=mem_id,
                                user_id=user.id,
                                app_id=app.id,
                                content=result['memory'],
                                state=MemoryState.active,
                                metadata_=original_metadata,
                            )
                            db.add(memory)
                        else:
                            memory.state = MemoryState.active
                            memory.content = result['memory']
                            memory.metadata_ = original_metadata
                        history = MemoryStatusHistory(
                            memory_id=mem_id,
                            changed_by=user.id,
                            old_state=(MemoryState.deleted if memory else None),
                            new_state=MemoryState.active,
                        )
                        db.add(history)
                    elif result['event'] == 'DELETE':
                        if memory:
                            memory.state = MemoryState.deleted
                            memory.deleted_at = datetime.datetime.now(datetime.timezone.utc)
                            history = MemoryStatusHistory(
                                memory_id=mem_id,
                                changed_by=user.id,
                                old_state=MemoryState.active,
                                new_state=MemoryState.deleted,
                            )
                            db.add(history)
                db.commit()
        finally:
            db.close()
    except Exception as bg_exc:
        logging.exception(f"Background add_memories failed: {bg_exc}")

@mcp.tool(description="Add a new memory, with optional user metadata and agent support.")
async def add_memories(text: str, metadata: Any = None, agent_id: Optional[str] = None) -> str:
    """
    Add a new memory. Accepts:
      - text: the memory content
      - metadata: optional key/value metadata to attach (must contain conversationId)
      - agent_id: optional agent ID to associate this memory with a specific agent
    """
    # Handle None metadata
    if metadata is None:
        metadata = {}
    
    # Extract user_id from metadata
    user_id = get_user_id_from_metadata(metadata)
    client_name = client_name_var.get(None)
    
    # Debug inputs
    logging.debug(f"add_memories called with text length={len(text) if isinstance(text, str) else 'n/a'}, metadata_keys={list(metadata.keys()) if isinstance(metadata, dict) else 'n/a'}, user_id={user_id}, client_name={client_name}, agent_id={agent_id}")
    
    if not user_id:
        return "Error: user_id not found in metadata.conversationId"
    if not client_name:
        return "Error: client_name not provided"

    memory_client = get_memory_client_safe()
    if not memory_client:
        return "Error: Memory system is currently unavailable. Please try again later."

    logger.debug(f"metadata after processing: {metadata}")
    logger.debug(f"Using user_id: {user_id}")
    logger.debug(f"Using agent_id: {agent_id}")
    
    # Merge default metadata
    full_meta = {
        **metadata,
        "source_app": "openmemory",
        "mcp_client": client_name,
    }
    
    # Add agent_id to metadata if provided
    if agent_id:
        full_meta["agent_id"] = agent_id
    
    logger.debug(f"full_meta to be sent to memory_client: {full_meta}")

    try:
        # Schedule background task and return immediately
        asyncio.create_task(
            _process_add_memories(
                text=text,
                full_meta=full_meta,
                agent_id=agent_id,
                user_id=user_id,
                client_name=client_name,
                original_metadata=metadata,
            )
        )
        return json.dumps({"status":"accepted"}, separators=(",", ":"))
    except Exception as e:
        logging.exception(f"Error scheduling background add_memories: {e}")
        return f"Error scheduling add_memories: {e}"


@mcp.tool(description="Search through stored memories. This method is called EVERYTIME the user asks anything.")
async def search_memory(query: str, metadata: Any = None, agent_id: Optional[str] = None) -> str:
    # Validate metadata presence
    if not isinstance(metadata, dict):
        return "Error: metadata must be a dictionary containing assistantId and collectionName"
    
    # Extract and validate required fields
    user_id = metadata.get("assistantId")
    collection_name = metadata.get("collectionName")
    client_name = client_name_var.get(None)
    
    # Validate required fields
    if not user_id:
        return "Error: assistantId not found in metadata - required for filtering"
    if not collection_name:
        return "Error: collectionName not found in metadata - required for filtering"
    if not client_name:
        return "Error: client_name not provided"

    logging.debug(f"search_memory called with query length={len(query)}, user_id={user_id}, collection_name={collection_name}, agent_id={agent_id}")

    # Get memory client safely
    memory_client = get_memory_client_safe()
    if not memory_client:
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        db = SessionLocal()
        try:
            # Get or create user and app using user_id from metadata
            user, app = get_user_and_app(db, user_id=user_id, app_id=client_name)

            # Get accessible memory IDs based on ACL
            user_memories = db.query(Memory).filter(Memory.user_id == user.id).all()
            accessible_memory_ids = [memory.id for memory in user_memories if check_memory_access_permissions(db, memory, app.id)]
            
            # Base conditions for Qdrant search - UPDATED with collectionName filtering
            conditions = [
                qdrant_models.FieldCondition(key="user_id", match=qdrant_models.MatchValue(value=user_id)),
                qdrant_models.FieldCondition(key="collectionName", match=qdrant_models.MatchValue(value=collection_name))
            ]
            
            # Add agent_id condition if provided
            if agent_id:
                conditions.append(qdrant_models.FieldCondition(key="agent_id", match=qdrant_models.MatchValue(value=agent_id)))
            
            if accessible_memory_ids:
                # Convert UUIDs to strings for Qdrant
                accessible_memory_ids_str = [str(memory_id) for memory_id in accessible_memory_ids]
                conditions.append(qdrant_models.HasIdCondition(has_id=accessible_memory_ids_str))

            filters = qdrant_models.Filter(must=conditions)
            # Offload potentially blocking embedding call
            embed_start = time.perf_counter()
            embeddings = await asyncio.to_thread(memory_client.embedding_model.embed, query, "search")
            logging.debug(f"embedding_model.embed completed in {time.perf_counter() - embed_start:.3f}s")
            
            # Offload vector store query
            query_start = time.perf_counter()
            hits = await asyncio.to_thread(
                memory_client.vector_store.client.query_points,
                collection_name=memory_client.vector_store.collection_name,
                query=embeddings,
                query_filter=filters,
                limit=10,
            )
            logging.debug(f"vector_store.query_points completed in {time.perf_counter() - query_start:.3f}s")

            # Process search results
            memories = hits.points
            memories = [
                {
                    "id": memory.id,
                    "memory": memory.payload["data"],
                    "hash": memory.payload.get("hash"),
                    "created_at": memory.payload.get("created_at"),
                    "updated_at": memory.payload.get("updated_at"),
                    "score": memory.score,
                    "collection_name": memory.payload.get("collectionName", ""),
                }
                for memory in memories
            ]

            # Log memory access for each memory found
            if isinstance(memories, dict) and 'results' in memories:
                print(f"Memories: {memories}")
                for memory_data in memories['results']:
                    if 'id' in memory_data:
                        memory_id = uuid.UUID(memory_data['id'])
                        # Create access log entry
                        access_log = MemoryAccessLog(
                            memory_id=memory_id,
                            app_id=app.id,
                            access_type="search",
                            metadata_={
                                "query": query,
                                "agent_id": agent_id,
                                "collection_name": collection_name,
                                "score": memory_data.get('score'),
                                "hash": memory_data.get('hash')
                            }
                        )
                        db.add(access_log)
                db.commit()
            else:
                for memory in memories:
                    memory_id = uuid.UUID(memory['id'])
                    # Create access log entry
                    access_log = MemoryAccessLog(
                        memory_id=memory_id,
                        app_id=app.id,
                        access_type="search",
                        metadata_={
                            "query": query,
                            "agent_id": agent_id,
                            "collection_name": collection_name,
                            "score": memory.get('score'),
                            "hash": memory.get('hash')
                        }
                    )
                    db.add(access_log)
                db.commit()
            
            logging.debug(f"Search returned {len(memories)} memories for collection '{collection_name}' and user '{user_id}'")
            # Return compact JSON string to avoid multi-line SSE payloads
            return json.dumps(memories, separators=(",", ":"))
        finally:
            db.close()
    except Exception as e:
        logging.exception(e)
        return f"Error searching memory: {e}"


@mcp.tool(description="List all memories in the user's memory")
async def list_memories(metadata: Any = None, agent_id: Optional[str] = None) -> str:
    # Validate metadata presence
    if not isinstance(metadata, dict):
        return "Error: metadata must be a dictionary containing assistantId and collectionName"
    
    # Extract and validate required fields
    user_id = metadata.get("assistantId")
    collection_name = metadata.get("collectionName")
    client_name = client_name_var.get(None)
    
    # Validate required fields
    if not user_id:
        return "Error: assistantId not found in metadata - required for filtering"
    if not collection_name:
        return "Error: collectionName not found in metadata - required for filtering"
    if not client_name:
        return "Error: client_name not provided"

    logging.debug(f"list_memories called with user_id={user_id}, collection_name={collection_name}, agent_id={agent_id}")

    # Get memory client safely
    memory_client = get_memory_client_safe()
    if not memory_client:
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=user_id, app_id=client_name)

            # Prepare parameters for get_all with collectionName filtering
            get_all_params = {
                "user_id": user_id,
                "metadata": {"collectionName": collection_name}
            }
            if agent_id:
                get_all_params["agent_id"] = agent_id

            # Get all memories
            # Offload blocking get_all
            get_all_start = time.perf_counter()
            memories = await asyncio.to_thread(memory_client.get_all, **get_all_params)
            logging.debug(f"memory_client.get_all completed in {time.perf_counter() - get_all_start:.3f}s")
            filtered_memories = []

            # Filter memories based on permissions
            user_memories = db.query(Memory).filter(Memory.user_id == user.id).all()
            accessible_memory_ids = [memory.id for memory in user_memories if check_memory_access_permissions(db, memory, app.id)]
            if isinstance(memories, dict) and 'results' in memories:
                for memory_data in memories['results']:
                    if 'id' in memory_data:
                        memory_id = uuid.UUID(memory_data['id'])
                        if memory_id in accessible_memory_ids:
                            # Create access log entry
                            access_log = MemoryAccessLog(
                                memory_id=memory_id,
                                app_id=app.id,
                                access_type="list",
                                metadata_={
                                    "agent_id": agent_id,
                                    "collection_name": collection_name,
                                    "hash": memory_data.get('hash')
                                }
                            )
                            db.add(access_log)
                            filtered_memories.append(memory_data)
                db.commit()
            else:
                for memory in memories:
                    memory_id = uuid.UUID(memory['id'])
                    memory_obj = db.query(Memory).filter(Memory.id == memory_id).first()
                    if memory_obj and check_memory_access_permissions(db, memory_obj, app.id):
                        # Create access log entry
                        access_log = MemoryAccessLog(
                            memory_id=memory_id,
                            app_id=app.id,
                            access_type="list",
                            metadata_={
                                "agent_id": agent_id,
                                "collection_name": collection_name,
                                "hash": memory.get('hash')
                            }
                        )
                        db.add(access_log)
                        filtered_memories.append(memory)
                db.commit()
            
            logging.debug(f"Listed {len(filtered_memories)} memories for collection '{collection_name}' and user '{user_id}'")
            # Return compact JSON string to avoid multi-line SSE payloads
            return json.dumps(filtered_memories, separators=(",", ":"))
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error getting memories: {e}")
        return f"Error getting memories: {e}"


@mcp.tool(description="Delete all memories in the user's memory")
async def delete_all_memories(metadata: Any = None, agent_id: Optional[str] = None) -> str:
    user_id = get_user_id_from_metadata(metadata)
    client_name = client_name_var.get(None)
    
    if not user_id:
        return "Error: user_id not found in metadata.conversationId"
    if not client_name:
        return "Error: client_name not provided"

    logging.debug(f"delete_all_memories called with user_id={user_id}, agent_id={agent_id}")

    # Get memory client safely
    memory_client = get_memory_client_safe()
    if not memory_client:
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=user_id, app_id=client_name)

            user_memories = db.query(Memory).filter(Memory.user_id == user.id).all()
            accessible_memory_ids = [memory.id for memory in user_memories if check_memory_access_permissions(db, memory, app.id)]

            # If agent_id is provided, filter memories by agent_id from metadata
            if agent_id:
                # Filter memories that belong to the specific agent
                agent_specific_memory_ids = []
                for memory_id in accessible_memory_ids:
                    memory_obj = db.query(Memory).filter(Memory.id == memory_id).first()
                    if memory_obj and memory_obj.metadata_ and memory_obj.metadata_.get("agent_id") == agent_id:
                        agent_specific_memory_ids.append(memory_id)
                accessible_memory_ids = agent_specific_memory_ids

            # delete the accessible memories only
            for memory_id in accessible_memory_ids:
                try:
                    memory_client.delete(memory_id)
                except Exception as delete_error:
                    logging.warning(f"Failed to delete memory {memory_id} from vector store: {delete_error}")

            # Update each memory's state and create history entries
            now = datetime.datetime.now(datetime.UTC)
            for memory_id in accessible_memory_ids:
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                # Update memory state
                memory.state = MemoryState.deleted
                memory.deleted_at = now

                # Create history entry
                history = MemoryStatusHistory(
                    memory_id=memory_id,
                    changed_by=user.id,
                    old_state=MemoryState.active,
                    new_state=MemoryState.deleted
                )
                db.add(history)

                # Create access log entry
                access_log = MemoryAccessLog(
                    memory_id=memory_id,
                    app_id=app.id,
                    access_type="delete_all",
                    metadata_={
                        "operation": "bulk_delete",
                        "agent_id": agent_id
                    }
                )
                db.add(access_log)

            db.commit()
            
            if agent_id:
                return f"Successfully deleted all memories for agent {agent_id}"
            else:
                return "Successfully deleted all memories"
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error deleting memories: {e}")
        return f"Error deleting memories: {e}"


@mcp_router.get("/{client_name}/sse/{user_id}")
async def handle_sse(request: Request):
    """Handle SSE connections for a specific user and client"""
    # Extract user_id and client_name from path parameters
    uid = request.path_params.get("user_id")
    user_token = user_id_var.set(uid or "")
    client_name = request.path_params.get("client_name")
    client_token = client_name_var.set(client_name or "")

    try:
        # Handle SSE connection
        logging.debug(f"Opening SSE connection for user_id={uid}, client_name={client_name}")
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            logging.debug("SSE connected; starting MCP server run")
            run_start = time.perf_counter()
            try:
                await mcp._mcp_server.run(
                    read_stream,
                    write_stream,
                    mcp._mcp_server.create_initialization_options(),
                )
            finally:
                logging.debug(f"MCP server run finished in {time.perf_counter() - run_start:.3f}s")
    finally:
        # Clean up context variables
        user_id_var.reset(user_token)
        client_name_var.reset(client_token)
        logging.debug(f"Closed SSE connection for user_id={uid}, client_name={client_name}")


@mcp_router.head("/{client_name}/sse/{user_id}")
async def handle_sse_head(request: Request):
    """Handle HEAD requests for SSE endpoint connectivity checks"""
    # Return successful response to indicate endpoint is available
    return {"status": "ok"}


@mcp_router.post("/messages/")
async def handle_get_message(request: Request):
    return await handle_post_message(request)


@mcp_router.post("/{client_name}/sse/{user_id}/messages/")
async def handle_post_message(request: Request):
    return await handle_post_message(request)

async def handle_post_message(request: Request):
    """Handle POST messages for SSE"""
    try:
        body = await request.body()
        headers = dict(request.headers)
        logging.debug(
            f"Received SSE POST message: path={request.url.path}, content_length={len(body)}, content_type={headers.get('content-type')}"
        )

        # Create a simple receive function that returns the body
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        # Create a simple send function that does nothing
        async def send(message):
            return {}

        # Call handle_post_message with the correct arguments
        post_start = time.perf_counter()
        await sse.handle_post_message(request.scope, receive, send)
        logging.debug(f"sse.handle_post_message completed in {time.perf_counter() - post_start:.3f}s")

        # Return a success response
        return {"status": "ok"}
    except Exception as e:
        logging.exception(f"Error handling SSE POST message: {e}")
        return {"status": "error", "detail": str(e)}
    finally:
        pass
        # Clean up context variable
        # client_name_var.reset(client_token)

def setup_mcp_server(app: FastAPI):
    """Setup MCP server with the FastAPI application"""
    mcp._mcp_server.name = f"mem0-mcp-server"

    # Include MCP router in the FastAPI app
    app.include_router(mcp_router)
