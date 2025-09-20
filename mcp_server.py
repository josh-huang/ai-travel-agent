from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

from dotenv import load_dotenv

try:
    from modelcontextprotocol import types as mcp_types
    from modelcontextprotocol.server import Server
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Install modelcontextprotocol[cli] to run the MCP server."
    ) from exc

from main import graph
from web_operations import serp_search

load_dotenv()

server = Server("travel-agent")


def _default_state(question: str) -> Dict[str, Any]:
    """Construct the baseline graph state for a new request."""
    return {
        "messages": [{"role": "user", "content": question}],
        "user_question": question,
        "google_results": None,
        "bing_results": None,
        "reddit_results": None,
        "selected_reddit_urls": None,
        "reddit_post_data": None,
        "google_analysis": None,
        "bing_analysis": None,
        "reddit_analysis": None,
        "final_answer": None,
    }


def _flight_lookup(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Use the existing SERP helper to fetch flight-related search results."""
    origin = arguments["origin"]
    destination = arguments["destination"]
    depart_date = arguments.get("depart_date")
    return_date = arguments.get("return_date")

    query_parts = [f"flights from {origin} to {destination}"]
    if depart_date:
        query_parts.append(f"on {depart_date}")
    if return_date:
        query_parts.append(f"return {return_date}")

    query = " ".join(query_parts)
    search_results = serp_search(query, engine="google")
    return {"query": query, "results": search_results}


async def _invoke_graph(question: str) -> Dict[str, Any]:
    """Run the LangGraph pipeline on a worker thread."""
    state = _default_state(question)
    return await asyncio.to_thread(graph.invoke, state)


@server.list_tools()
async def list_tools() -> list[mcp_types.Tool]:
    """Expose the travel tools to MCP clients."""
    return [
        mcp_types.Tool(
            name="travel.plan_trip",
            description=(
                "Run the full travel research agent and return a synthesized itinerary."
            ),
            input_schema=mcp_types.JSONSchema(
                type="object",
                properties={
                    "question": mcp_types.JSONSchema(type="string", description="User travel request"),
                },
                required=["question"],
            ),
        ),
        mcp_types.Tool(
            name="travel.search_flights",
            description=(
                "Perform a targeted flight availability search using Bright Data SERP." \
                " Returns the parsed search response for downstream tooling."
            ),
            input_schema=mcp_types.JSONSchema(
                type="object",
                properties={
                    "origin": mcp_types.JSONSchema(type="string", description="IATA origin code"),
                    "destination": mcp_types.JSONSchema(type="string", description="IATA destination code"),
                    "depart_date": mcp_types.JSONSchema(type="string", description="YYYY-MM-DD departure date"),
                    "return_date": mcp_types.JSONSchema(type="string", description="Optional return date"),
                },
                required=["origin", "destination"],
            ),
        ),
    ]


@server.call_tool()
async def call_tool(
    name: str, arguments: Dict[str, Any]
) -> mcp_types.ToolResponse:
    """Route MCP tool invocations to the underlying agent functions."""
    if name == "travel.plan_trip":
        question = arguments["question"]
        result = await _invoke_graph(question)
        final_answer = result.get("final_answer") or "No response produced."
        payload = json.dumps(result, indent=2)
        return mcp_types.ToolResponse(
            content=[
                mcp_types.TextContent(text=final_answer),
                mcp_types.BlobContent(
                    data=payload.encode("utf-8"),
                    media_type="application/json",
                    filename="plan_trip.json",
                ),
            ]
        )

    if name == "travel.search_flights":
        flight_payload = await asyncio.to_thread(_flight_lookup, arguments)
        summary = flight_payload["query"]
        payload = json.dumps(flight_payload, indent=2)
        return mcp_types.ToolResponse(
            content=[
                mcp_types.TextContent(
                    text=f"Flight SERP query executed: {summary}"
                ),
                mcp_types.BlobContent(
                    data=payload.encode("utf-8"),
                    media_type="application/json",
                    filename="flight_search.json",
                ),
            ]
        )

    raise mcp_types.ToolError(f"Unsupported tool: {name}")


def run() -> None:
    """Launch the MCP server."""
    server.run()


if __name__ == "__main__":
    run()
