"""
OpenAI built-in web search helper using the Responses API.

Pattern per request:
from openai import OpenAI
client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    tools=[{"type": "web_search"}],
    input="What was a positive news story from today?"
)
print(response.output_text)
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

        # Use Responses API directly with plain string input and web_search tool
        resp = self.client.responses.create(
            model="gpt-5",
            tools=[{"type": "web_search"}],
            input=f"{query}",
            max_output_tokens=300,
            temperature=0.3,
        )

        # Return a single summarized item; callers expect a list
        text = getattr(resp, "output_text", "") or ""
        return [{"title": "web search summary", "url": "", "snippet": text}]


