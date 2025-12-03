from __future__ import annotations

from typing import Any, Dict, List

from openai import AsyncOpenAI


class GroqAdapter:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile") -> None:
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        self.model = model

    async def chat(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, *messages],
            temperature=0.3,
        )
        return response.choices[0].message.content or ""


__all__ = ["GroqAdapter"]
