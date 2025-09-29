"""
OpenAI built-in web search helper using the Responses API.

Requires OPENAI_API_KEY in environment. Uses model 'gpt-4.1' with tools=[{"type": "web_search"}].
"""

import os
from typing import List, Dict, Any
from openai import OpenAI


class OpenAIWebSearch:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        if not self.client or not query:
            return []

        # Minimal input conforming to Responses API with text content
        input_msgs = [
            {"role": "user", "content": [{"type": "text", "text": f"Search the web: {query}"}]}
        ]

        resp = self.client.responses.create(
            model="gpt-4.1",
            input=input_msgs,
            tools=[{"type": "web_search"}],
            tool_choice="auto",
            max_output_tokens=300,
            temperature=0.3,
        )

        # The Responses API returns a synthesized answer; we return a single item with text
        text = getattr(resp, "output_text", "") or ""
        return [{"title": "web search summary", "url": "", "snippet": text}]


