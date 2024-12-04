from fastapi import FastAPI
from agents.rag_agent import salt_cfp_rag_agent_executor
from models.query import QueryInput, QueryOutput
from retrievers.document_retriever import db
from utils.async_utils import async_retry


app = FastAPI(
    title="SALT Chatbot",
    description="Endpoints for a SALT chatbot"
)

@async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str):
    """Retry the agent if a tool fails to run.

    This can help when there are intermittent connection issues to external APIs.
    """
    return await salt_cfp_rag_agent_executor.ainvoke({"input": query})

@app.get("/")
async def get_status():
    return {"status": "running"}

@app.get("/salt-rag-agent")
async def query_salt_agent(query: QueryInput) -> QueryOutput:
    query_response = await invoke_agent_with_retry(query.text)
    # Format sources to include header and link
    raw_sources = query_response.get("sources", [])
    data = db.get()
    headers = [metadata["header"] for metadata in data["metadatas"]]
    # print(headers)
    formatted_sources = []
    for source in raw_sources:
        formatted_sources.append(headers[source])

    # Extract and format intermediate steps
    intermediate_steps = [
        str(step) for step in query_response.get("intermediate_steps", [])
    ]

    # Construct the response object
    return QueryOutput(
        answer=query_response.get("answer", ""),
        sources=formatted_sources,
        intermediate_steps=intermediate_steps
    )