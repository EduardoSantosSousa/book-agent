import asyncio
import time
from typing import List, Dict
from groq import Groq

class GroqService:
    def __init__(self, model: str, api_key: str, timeout: int = 60):
        self.model = model
        self.client = Groq(api_key=api_key)
        self.timeout = timeout
        self.response_times = []

    async def chat(self, messages: List[Dict[str, str]]) -> str:
        start = time.time()

        def _call():
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return resp.choices[0].message.content

        result = await asyncio.to_thread(_call)
        self.response_times.append(time.time() - start)
        return result

    async def health_check(self) -> bool:
        return True
