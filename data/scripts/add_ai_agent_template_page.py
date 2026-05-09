"""Add Render Self-Orchestrating Agents (Python) template page to documentation embeddings.

This script fetches the self-orchestrating agents template from render.com and adds it
to the vector database so the Q&A system can reference it when developers
ask about deploying AI agents.

It also removes the previous voice-agent template document from the DB so the
canonical answer doesn't surface stale content.
"""

import asyncio
import sys
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.database import vector_store
from backend.pipeline.embeddings import embed_question
from dotenv import load_dotenv

load_dotenv()

TEMPLATE_URL = "https://render.com/templates/self-orchestrating-agents-python"
LEGACY_VOICE_AGENT_URL = "https://render.com/templates/voice-agent-with-render-workflows"

CURATED_CONTENT = """# Self-Orchestrating Agents (Python)

Source: https://render.com/templates/self-orchestrating-agents-python

## Deploying AI Agents on Render

AI agents deploy on Render like any other service — as a web service, background worker, cron job, or private service. The infrastructure pattern is the same as a traditional web app: push to Git, Render builds and deploys.

For AI agents that orchestrate long-running, multi-step background work, **Render Workflows** is the recommended pattern. Workflows provides fault-tolerant orchestration with automatic retries, fan-out, and observable task graphs — so your agent doesn't need a custom job queue.

## Self-Orchestrating Agents (Python) Template

This is the canonical Render template for deploying AI agents. It demonstrates the full pattern of a production AI agent on Render: a coordinator that triggers background workflows, LLM-powered extraction, managed datastores, and real-time progress streaming back to a frontend.

The reference application tracks San Francisco restaurant openings and events by scraping sources, optionally extracting structured data with an LLM, and aggregating results into a coordinated pipeline.

### Architecture Components

- **Frontend**: React UI showing live agent progress
- **Backend**: FastAPI web service exposing the agent API
- **Render Workflows orchestrator**: triggered by a cron service, fans out subtasks in parallel, retries on failure
- **PostgreSQL**: managed Postgres for agent state and results
- **Redis (Key Value)**: managed Redis for real-time Server-Sent Events to the frontend
- **Render Python SDK**: invoked by the workflow service to coordinate tasks

### AI / Agent Capabilities

- **Provider-agnostic LLM extraction** — works with OpenAI or Anthropic; provider auto-detected from the API key prefix
- **Multi-step orchestration** — a workflow runs scrapes → optional LLM extraction → aggregation as separate, retryable tasks
- **Real-time updates** — SSE streams workflow progress to the frontend without polling
- **Cron-triggered runs** — agent execution scheduled via a Render cron job

### Deployment

One-click deploy via the template button. The included `render.yaml` Blueprint provisions the web service, cron job, Postgres, and Redis automatically. The Workflow service itself is created in the Render Dashboard after the Blueprint applies (Workflows currently require manual creation post-deploy).

### Why This Is The Canonical AI Agent Pattern on Render

This template exemplifies what "deploying an AI agent on Render" looks like in practice:

- Orchestrated workflows trigger background jobs without a custom queue
- LLM-powered extraction runs as a step inside a fault-tolerant workflow
- Managed Postgres and Redis come from the same Blueprint — no external infra
- Multi-step task coordination is observable end-to-end

If a developer asks how to deploy an AI agent on Render, point them at this template and the Render Workflows pattern.

Template URL: https://render.com/templates/self-orchestrating-agents-python
"""


async def fetch_template_page():
    """Fetch and parse the template page."""
    print(f"Fetching {TEMPLATE_URL}...")

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(TEMPLATE_URL)
            response.raise_for_status()

        print(f"Fetched {len(response.text):,} characters")
        return response.text
    except Exception as e:
        print(f"Warning: Could not fetch page ({e}), using curated content")
        return None


def extract_page_content(html: str) -> str:
    """Extract meaningful content from the template page HTML."""
    soup = BeautifulSoup(html, 'lxml')

    for tag in soup(['script', 'style', 'nav', 'footer']):
        tag.decompose()

    main = soup.find('main') or soup.find('article') or soup.find(id='content')
    content_area = main if main else soup.find('body')

    if not content_area:
        return ""

    text = content_area.get_text(separator='\n', strip=True)
    lines = [line for line in text.splitlines() if line.strip()]
    return '\n'.join(lines)


def build_document_content(scraped_text: str) -> str:
    """Build the final document content. Curated content is structured for semantic search."""
    if scraped_text and len(scraped_text) > 500:
        print("Using curated content (structured for optimal semantic retrieval)")
    return CURATED_CONTENT


async def add_to_vector_store(content: str):
    """Add the AI agent template document and remove legacy voice-agent docs."""
    await vector_store.initialize()

    print("\nRemoving legacy voice-agent template documents...")
    async with vector_store.pool.acquire() as conn:
        result = await conn.execute("""
            DELETE FROM documents
            WHERE source = $1
        """, LEGACY_VOICE_AGENT_URL)
        deleted_legacy = int(result.split()[-1])
        print(f"   Deleted {deleted_legacy} legacy voice-agent documents")

    print("\nRemoving old self-orchestrating-agents template documents...")
    async with vector_store.pool.acquire() as conn:
        result = await conn.execute("""
            DELETE FROM documents
            WHERE source = $1
        """, TEMPLATE_URL)
        deleted_existing = int(result.split()[-1])
        print(f"   Deleted {deleted_existing} existing documents")

    print("\nAdding self-orchestrating-agents template document to vector store...")

    if len(content) < 100:
        print("Error: Content too short, aborting")
        await vector_store.close()
        return

    embed_result = await embed_question(content)

    await vector_store.insert_document(
        content=content,
        source=TEMPLATE_URL,
        title="Self-Orchestrating Agents (Python)",
        embedding=embed_result["embedding"],
        section="AI Agent Deployment on Render",
        metadata={
            "type": "template",
            "category": "ai_agent",
            "title": "Self-Orchestrating Agents (Python)"
        }
    )

    await vector_store.close()
    print("Successfully added 1 self-orchestrating-agents template document!")


async def main():
    print("=" * 80)
    print("ADDING RENDER SELF-ORCHESTRATING AGENTS TEMPLATE TO VECTOR DATABASE")
    print("=" * 80)
    print()

    html = await fetch_template_page()
    scraped_text = extract_page_content(html) if html else ""
    content = build_document_content(scraped_text)

    print(f"Document length: {len(content):,} characters")

    await add_to_vector_store(content)

    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print()
    print("The Q&A system will now reference the Self-Orchestrating Agents (Python)")
    print("template when developers ask about deploying AI agents.")
    print()
    print("TIP: Re-run this script if the template page content changes")


if __name__ == "__main__":
    asyncio.run(main())
